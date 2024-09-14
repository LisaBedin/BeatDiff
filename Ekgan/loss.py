import torch
import torch.nn as nn
import torch.nn.functional as F

def label_generator_loss(lg_output, input_image):
    target_loss = torch.mean(torch.abs(input_image - lg_output))
    total_disc_loss = target_loss
    return total_disc_loss, target_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    criterion = nn.BCEWithLogitsLoss()
    real_labels = torch.ones_like(disc_real_output)
    generated_labels = torch.zeros_like(disc_generated_output)

    real_loss = criterion(disc_real_output, real_labels)
    generated_loss = criterion(disc_generated_output, generated_labels)

    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def inference_generator_loss(disc_generated_output, ig_output, target, lambda_, ig_lv, lg_lv, alpha):
    criterion = nn.BCEWithLogitsLoss()
    gan_loss = criterion(disc_generated_output, torch.ones_like(disc_generated_output))
    l1_loss = F.l1_loss(ig_output, target)
    vector_loss = torch.mean(torch.abs(ig_lv - lg_lv))

    total_gen_loss = l1_loss * lambda_ + gan_loss + vector_loss * alpha

    return total_gen_loss, gan_loss, l1_loss, vector_loss