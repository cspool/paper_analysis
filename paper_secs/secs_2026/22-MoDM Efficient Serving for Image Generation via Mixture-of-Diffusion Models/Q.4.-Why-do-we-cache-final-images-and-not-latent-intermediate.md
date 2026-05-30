# Q.4. Why do we cache final images and not latent intermediate?

**Response:** Final images are directly usable and *model in-dependent*, making them universally compatible across all model families. In contrast, intermediate latents vary between models, limiting serving to a single model.

