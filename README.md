# sd-webui-freeu
implementation of [FreeU](https://github.com/ChenyangSi/FreeU) as an [a1111 sd webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) extension

# Settings

- Start At Step: do not apply FreeU until this sampling step is reached
- Stop At Step: apply FreeU until this sampling step is reached
- Transition Smoothness: see $k_{smooth}$ in [this desmos graph](https://www.desmos.com/calculator/6dwhr2r4ta)
- Backbone n Scale: scalar applied to the backbone window during UNet stage n
- Backbone n Offset: offset of the window, 1 is the same as 0 as the window wraps around the downsampled latent features
- Backbone n Width: width of the window applied to the backbone
- Skip n Scale: scalar applied to the low frequencies (low end) of the skip connection during UNet stage n
- Skip n High End Scale: scalar applied to the high frequencies (high end) of the skip connection
- Skip n Cutoff: ratio that separates low from high frequencies, 0 means single out the lowest frequency and 1 means scale all frequencies with "Skip n Scale"
