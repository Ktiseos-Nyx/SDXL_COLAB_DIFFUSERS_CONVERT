# SDXL_COLAB_DIFFUSERS_CONVERT
At your own risk, SDXL safetensors to Diffusers.

# **SDXL Model Converter Colab Edition**

> ## **⚠️ IMPORTANT: Google Colab AUP Warning & Xformers Disclaimer ⚠️**
---
This Google Colab notebook is designed to convert Stable Diffusion XL (SDXL) model checkpoints to the `diffusers` format. It uses code from the `kohya-ss/sd-scripts` repository, which is primarily designed for training Stable Diffusion models and may cause the notebook to violate the Google Colab Acceptable Use Policy (AUP), even if you are not actively training a model.

**Therefore, using this notebook carries a significant risk of AUP violations and may lead to the suspension of your Google Colab account. Use at your own risk.**

Furthermore, the `xformers` library, which is installed as part of this notebook, may not be compatible with all hardware configurations or versions of PyTorch, and may cause errors or unexpected behavior. The reliability of this code and the functionality of `xformers` is entirely at your own risk.

**This Notebook is Provided "AS-IS" with No Warranty or Guarantee of Functionality or Support.**


---
> ## Collaboration

>I am NOT A programmer by nature, I patch with what little knowledge I have. I Failed programming several times over the years, so if something needs cleaning up and you want to patch it - pull request it!
---
>## About


>We are a system of over 300 alters, proudly navigating life with Dissociative Identity Disorder, ADHD, Autism, and CPTSD. We believe in the potential of AI to break down barriers and enhance aspects of mental health, even as it presents challenges. Our creative journey is an ongoing exploration of identity and expression, and we invite you to join us in this adventure.

---


>## Credits:


| Patched Origin | Description | Link |
| --- | --- | --- |
|Patched from| ARCHIVED |[SDXL - Linaqruf](https://colab.research.google.com/github/Linaqruf/sdxl-model-converter/blob/main/sdxl_model_converter.ipynb)
|***Linaqruf @ Github***: |https://github.com/Linaqruf
|Linaqruf Ko-Fi | [![](https://dcbadge.vercel.app/api/shield/850007095775723532?style=flat)](https://lookup.guru/850007095775723532) [![ko-fi](https://img.shields.io/badge/Support%20me%20on%20Ko--fi-F16061?logo=ko-fi&logoColor=white&style=flat)](https://ko-fi.com/linaqruf)
| Linaqruf Saweria |<a href="https://saweria.co/linaqruf"><img alt="Saweria" src="https://img.shields.io/badge/Saweria-7B3F00?style=flat&logo=ko-fi&logoColor=white"/></a>
>## Social Media
---
| Link Name| Description | Link |
| --- | --- | --- |
| [Huggingface Backup](https://colab.research.google.com/github/kieranxsomer/HuggingFace_Backup/blob/main/HuggingFace_Backup.ipynb) | backup checkpoints! | [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://colab.research.google.com/github/kieranxsomer/HuggingFace_Backup/blob/main/HuggingFace_Backup.ipynb)
|Discord| E&D Discord |[Invite](https://discord.gg/5t2kYxt7An)
|CivitAi| Duskfallcrew @ Civitai |[Duskfallcrew](https://civitai.com/user/duskfallcrew/)
|Huggingface| E&D Huggingface |[Earth & Dusk](https://huggingface.co/EarthnDusk)
|Ko-Fi| Kofi Support |[![ko-fi](https://img.shields.io/badge/Support%20me%20on%20Ko--fi-F16061?logo=ko-fi&logoColor=white&style=flat)](https://ko-fi.com/Z8Z8L4EO)
|Github| Duskfallcrew Github |[Duskfallcrew](https://github.com/duskfallcrew)
| Youtube: | Duskfall Music|[Duskfall Music & More](https://www.youtube.com/channel/UCk7MGP7nrJz5awBSP75xmVw)
| Spotify: | E&D Royalty Free| [PLAYLIST](https://open.spotify.com/playlist/00R8x00YktB4u541imdSSf?si=57a8f0f0fe87434e)
|DA Group | AI Group| [DeviantArt Group](https://www.deviantart.com/diffusionai)
| Reddit | Earth & Dusk| [Subreddit](https://www.reddit.com/r/earthndusk/)

---
