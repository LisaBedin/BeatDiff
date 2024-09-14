"""
PyTorch model for the uncertainty estimation.

Copyright 2019
Vincent Fortuin
Microsoft Research Cambridge
"""

from collections import namedtuple, OrderedDict
import torch
import torch.nn as nn

# ================== ================= #

# ================== ================= #

# ============== some utils ============= #
union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}

def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, (*pfx, name))
        else: yield ((*pfx, name), val)

sep = '_'
RelativePath = namedtuple('RelativePath', ('parts'))
rel_path = lambda *parts: RelativePath(parts)


def build_graph(net):
    net = dict(path_iter(net))
    default_inputs = [[('input',)]] + [[k] for k in net.keys()]
    with_default_inputs = lambda vals: (val if isinstance(val, tuple) else (val, default_inputs[idx]) for idx, val in
                                        enumerate(vals))
    parts = lambda path, pfx: tuple(pfx) + path.parts if isinstance(path, RelativePath) else (path,) if isinstance(path,
                                                                                                                   str) else path
    return {sep.join((*pfx, name)): (val, [sep.join(parts(x, pfx)) for x in inputs]) for (*pfx, name), (val, inputs) in
            zip(net.keys(), with_default_inputs(net.values()))}

def scale_prior(model, scaling_factor):
    old_params = model.prior.state_dict()
    new_params = OrderedDict()

    for k, v in old_params.items():
        if k.split(".")[-1] in ["weight", "bias"]:
            new_params[k] = v * scaling_factor
        else:
            new_params[k] = v

    model.prior.load_state_dict(new_params)


# =================== base network class ============== #
class Network(nn.Module):
    def __init__(self, net):
        self.graph = build_graph(net)
        super().__init__()
        for n, (v, _) in self.graph.items():
            setattr(self, n, v)

    def forward(self, inputs):
        self.cache = dict(inputs)
        for n, (_, i) in self.graph.items():
            self.cache[n] = getattr(self, n)(*[self.cache[x] for x in i])
        return self.cache

    def half(self):
        for module in self.children():
            if not isinstance(module, nn.BatchNorm1d):
                module.half()
        return self


# ====================== model utils ================== #
def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False, bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm1d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False

    return m

def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv1d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw),
        'relu': nn.ReLU(True)
    }


def residual(c, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')]),
    }


class Identity(nn.Module):
    def forward(self, x): return x


class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, x):
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)  #x.size(1))


class Add(nn.Module):
    def forward(self, x, y): return x + y


# =================== building network function =================== #
def basic_net(c_in, channels, weight, pool, num_outputs=10, **kw):
    return {
        'prep': conv_bn(c_in, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),

        'pool': nn.MaxPool1d(4),
        'flatten': Flatten()
        #'linear': nn.Linear(2560, num_outputs), # channels['layer3']*5, num_outputs),
        #'out': Mul(weight),
    }

def large_net(c_in, channels, weight, pool, num_outputs=10, **kw):
    return {
        'prep': conv_bn(c_in, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'layer4': dict(conv_bn(channels['layer3'], channels['layer4'], **kw)),
        'layer5': dict(conv_bn(channels['layer4'], channels['layer5'], **kw)),

        'pool': nn.MaxPool2d(4),
        'flatten': Flatten()
        #'linear': nn.Linear(640, num_outputs),
        #'out': Mul(weight),
    }


def net(c_in=9, channels=None, weight=1.0, pool=nn.MaxPool1d(2), extra_layers=(), output_size=10,
        res_layers=('layer1', 'layer3'), net_type="basic", **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512, 'layer4': 512, 'layer5': 512}
    if net_type == "basic":
        n = basic_net(c_in, channels, weight, pool, num_outputs=output_size, **kw)
    elif net_type == "large":
        n = large_net(c_in, channels, weight, pool, num_outputs=output_size, **kw)
    #else:
    #    raise ValueError("Unknown net_type.")
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)
    return n

def head(num_outputs, weight=1.0, net_type='basic', joint=False):
    if net_type == 'basic':
        input_dim = 2560
        if joint:
            input_dim += 4096
    elif net_type == 'large':
        input_dim = 640
        # if joint:
        #    input_dim += 1024
    return nn.Sequential(*[nn.Linear(input_dim, num_outputs), # channels['layer3']*5, num_outputs),
                           Mul(weight)]).half()

def embedder(emb_channels, out_dim, net_type='basic'):
    if net_type == 'basic':
        return nn.Sequential(*[nn.Linear(4, 2 * emb_channels),
                               nn.SiLU(),
                               nn.Linear(2 * emb_channels, 2*emb_channels),
                               nn.AvgPool1d(2),
                               nn.Linear(emb_channels, out_dim)]).half()
    elif net_type == 'large':
        return nn.Sequential(*[nn.Linear(4, 2 * emb_channels),
                               nn.SiLU(),
                               nn.Linear(2 * emb_channels, 2*emb_channels),
                               nn.AvgPool1d(2),
                               nn.Linear(emb_channels, emb_channels),
                               nn.SiLU(),
                               nn.Linear(emb_channels, emb_channels),
                               nn.AvgPool1d(2),
                               nn.Linear(int(emb_channels//2), out_dim)]).half()



# =================== global model ================== #
class TorchUncertaintyPair(nn.Module):
    def __init__(self, c_in=9, uncertainty_threshold=1e-2, output_size=10, output_weight=1.,
                 init_scaling=1., gp_weight=1., conditioned=False):
        super().__init__()
        self.c_in = c_in
        self.learner = Network(net(c_in, net_type="large", output_size=output_size, weight=output_weight))
        self.prior = Network(net(c_in, net_type="basic", output_size=output_size, weight=output_weight))
        self.head_learner = head(output_size, output_weight, 'large', conditioned)# TODO
        self.conditioned = conditioned
        self.head_prior = head(output_size, output_weight, 'basic', conditioned)# TODO
        # self.embedder_learner = embedder(4096, 1024, 'large') # TODO, to embed features into a representation space
        self.embedder_prior = embedder(4096, 4096, 'basic') # TODO
        scale_prior(self, init_scaling)
        for param in self.prior.parameters():
            param.requires_grad = False
        self.uncertainty_threshold = uncertainty_threshold
        self.gp_weight = gp_weight

    def forward(self, inputs):
        prior_output = self.prior(inputs)['flatten']
        learner_output = self.learner(inputs)['flatten']
        if self.conditioned:
            prior_emb = self.embedder_prior(inputs['feats'])
            #learner_emb = self.embedder_learner(inputs['feats'])
            prior_output = torch.cat([prior_output, prior_emb], dim=-1)
            #learner_output = torch.cat([learner_output, learner_emb], dim=-1)
        prior_output = self.head_prior(prior_output)
        learner_output = self.head_learner(learner_output)
        diff = (prior_output - learner_output)
        squared_diff = diff * diff
        msd = torch.mean(squared_diff, dim=-1)
        correct = msd < self.uncertainty_threshold
        loss = msd
        if self.gp_weight > 0.:
            inputs_var = torch.autograd.Variable(inputs['input'], requires_grad=True)
            outputs_var = torch.mean(self.learner({"input": inputs_var})['out'])
            grads = torch.autograd.grad(outputs_var, inputs_var, create_graph=True,
                                        retain_graph=True, only_inputs=True)[0]
            grads_norm = grads.norm()
            loss = loss + self.gp_weight * grads_norm
        return {'uncertainties': msd, 'loss': loss, 'correct': correct}


class TorchUncertaintyEnsemble(nn.Module):
    def __init__(self, c_in=9, ensemble_size=2, uncertainty_threshold=1e-2, output_size=10,
                 output_weight=1., init_scaling=1., gp_weight=0., beta=1., conditioned=False):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.c_in = c_in
        self.conditioned = conditioned
        self.beta = beta
        self.uncertainty_threshold = uncertainty_threshold
        self.ensemble = []
        for i in range(ensemble_size):
            setattr(self, f"pair_{i}", TorchUncertaintyPair(c_in=c_in,
                                                            uncertainty_threshold=uncertainty_threshold,
                                                            output_size=output_size,
                                                            output_weight=output_weight,
                                                            init_scaling=init_scaling,
                                                            gp_weight=gp_weight,
                                                            conditioned=conditioned))
            self.ensemble.append(getattr(self, f"pair_{i}"))

    def forward(self, inputs):
        outputs = [pair(inputs) for pair in self.ensemble]
        losses_stacked = torch.stack([out['loss'] for out in outputs])
        losses_mean = torch.sum(losses_stacked, axis=0)
        uncertainties_stacked = torch.stack([out['uncertainties'] for out in outputs])
        if self.ensemble_size >= 2:
            uncertainties_combined = torch.mean(uncertainties_stacked, axis=0) + self.beta * torch.std(
                uncertainties_stacked, axis=0)
        else:
            uncertainties_combined = torch.mean(uncertainties_stacked, axis=0)
        correct = uncertainties_combined < self.uncertainty_threshold
        return {'uncertainties': uncertainties_combined, 'loss': losses_mean, 'correct': correct}