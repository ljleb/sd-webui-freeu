# sd-webui-freeu
implementation of [FreeU](https://github.com/ChenyangSi/FreeU) as an [a1111 sd webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) extension

At each of the 3 stages of the UNet decoder:
- Apply a scalar on a window over the features of the backbone
- Tame the frequencies of the skip connection

## Settings

- Start At Step: do not apply FreeU until this sampling step is reached
- Stop At Step: apply FreeU until this sampling step is reached
- Transition Smoothness: see $k_{smooth}$ in [this desmos graph](https://www.desmos.com/calculator/ngcqo5ictm)
- Backbone n Scale: scalar applied to the backbone window during UNet stage n
- Backbone n Offset: offset of the window, 1 is the same as 0 as the window wraps around the downsampled latent features
- Backbone n Width: width of the window applied to the backbone
- Skip n Scale: scalar applied to the low frequencies (low end) of the skip connection during UNet stage n
- Skip n High End Scale: scalar applied to the high frequencies (high end) of the skip connection
- Skip n Cutoff: ratio that separates low from high frequencies, 0 means to control the single lowest frequency with "Skip n Scale" and 1 means scale all frequencies with "Skip n Scale"

## API

You can pass a single dict as the alwayson script args when making API calls:

```json
{
    "alwayson_scripts": {
        "freeu": {
            "args": [{
                "enable": true,
                "start_ratio": 0.1,
                "stop_ratio": 0.9,
                "transition_smoothness": 0.1,
                "stage_infos": [
                    {
                        "backbone_factor": 1.2,
                        "backbone_offset": 0.5,
                        "backbone_width": 0.75,
                        "skip_factor": 0.9,
                        "skip_high_end_factor": 1.1,
                        "skip_cutoff": 0.3
                    },
                    {
                        "backbone_factor": 1.4,
                        "backbone_offset": 0.5,
                        "backbone_width": 0.75,
                        "skip_factor": 0.2,
                        "skip_high_end_factor": 1.1,
                        "skip_cutoff": 0.3
                    },
                    {
                        "backbone_factor": 1.1,
                        "backbone_offset": 0.5,
                        "backbone_width": 0.75,
                        "skip_factor": 0.9,
                        "skip_high_end_factor": 1.1,
                        "skip_cutoff": 0.3
                    }
                ]
            }]
        }
    }
}
```

It is possible to omit any of the entries. For example:

```json
{
    "alwayson_scripts": {
        "freeu": {
            "args": [{
                "start_ratio": 0.1,
                "stage_infos": [
                    {
                        "backbone_factor": 0.8,
                        "backbone_offset": 0.5,
                        "skip_high_end_factor": 0.9
                    }
                ]
            }]
        }
    }
}
```

Here, since there is a single dict in the `stage_infos` array, freeu will only have an effect during the first stage of the unet.  
If you want to modify only the second stage, prepend the `"stage_infos"` array with 1 empty dict `{}`.  
If you want to modify only the third stage, prepend the `"stage_infos"` array with 2 empty dicts.

If `"stop_ratio"` or `"start_ratio"` is an integer, then it is a step number.  
Otherwise, it is expected to be a float between `0.0` and `1.0` and it represents a ratio of the total sampling steps.
