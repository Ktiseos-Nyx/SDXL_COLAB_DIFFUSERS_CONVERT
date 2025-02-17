# **SDXL Model Converter Colab Edition**
---
## **⚠️IMPORTANT: Google Colab AUP Warning & Xformers Disclaimer⚠️**
---
 **⚠️ WARNING:**
This Google Colab notebook includes tools from the `kohya-ss/sd-scripts` repository, which is **primarily designed for training Stable Diffusion models**. Using this code may cause your Google Colab account to be flagged for violating the Google Colab Acceptable Use Policy (AUP), **even if you are not actively training a model**.

**⚠️ IMPORTANT AUP RISK:**
 Using this notebook carries a **significant risk of AUP violations** which may lead to the suspension of your Google Colab account and it is provided **AS-IS**  While it's not explicitly stated within the AUP, nor terms of service, slowdowns and over use on the free plans could lead to account strikes or removal. Please be aware that Ktiseos Nyx, Duskfallcrew, and even Google are not at fault for what pranks you try and pull with the AUP.

 **⚠️ XFORMERS WARNING:**
 The `xformers` library may not be compatible with all hardware configurations or versions of PyTorch, and it may lead to errors. The use of `xformers` and the reliability of this code is entirely at your own risk.

 **⚠️GRADIO AND PYTHON⚠️**
I am not a natural programmer. I have yet to test all of the extra files included except for the one that came from Linaqruf's original repository. If there is an issue feel free to open an issue case and i'll try and solve it.  Gradio MAY or may not work, i dont know how that works yet give me time.

 Be patient. Be kind. Rewind.

 I'm working on a re-do of the gradio!


---


## Links

| Link Name| Description | Link |
| --- | --- | --- |
| Huggingface Backup| backup checkpoints! | [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://github.com/Ktiseos-Nyx/HuggingFace_Backup)
|Discord| E&D Discord |[Invite](https://discord.gg/5t2kYxt7An)
|Huggingface| E&D Huggingface |[Earth & Dusk](https://huggingface.co/EarthnDusk)
|Ko-Fi| Kofi Support |[![ko-fi](https://img.shields.io/badge/Support%20me%20on%20Ko--fi-F16061?logo=ko-fi&logoColor=white&style=flat)](https://ko-fi.com/Z8Z8L4EO)
|Github| Ktiseos Nyx |[Ktiseos Nyx](https://github.com/Ktiseos-Nyx/)




## **Important Note:**

Before diving in, ensure you create a Hugging Face token with write permissions. Follow this link for instructions on token creation.

You need to create a huggingface token, go to [this link](https://huggingface.co/settings/tokens), then `create new token` or copy available token with the `Write` role.


### **Setup Instructions:**

1.  **Google Colab Environment:** This notebook is designed to run on Google Colab with a GPU runtime enabled.
2.  **Root Directory:** The `root_dir` variable specifies where files will be stored. The default is `/content`. You can specify a different location by changing the value in the corresponding text field.
3. **Repository URL:** If you want to use a specific version of the `kohya-ss/sd-scripts` repository, then you can specify the URL using the text field for "Repo URL".
4. **Branch Name:** If you want to use a specific branch of the `kohya-ss/sd-scripts` repository, then you can specify the branch name using the text field for "Branch".
5. **Xformers URL:** You are using the official xformers library, however, if you have a custom built `xformers` wheel file, then you can specify the URL using the "Xformers URL" field. (Not Recommended).
6.  **Run Setup:** After providing the values, click the "Setup Environment" button to install all required dependencies and clone the necessary files, and prepare the environment variables.
7.  **Follow the Steps:** You are now able to continue with the following steps.



>## Credits:


| Patched Origin | Description | Link |
| --- | --- | --- |
|Patched from| ARCHIVED |[SDXL - Linaqruf](https://colab.research.google.com/github/Linaqruf/sdxl-model-converter/blob/main/sdxl_model_converter.ipynb)
|***Linaqruf @ Github***: |https://github.com/Linaqruf
|Linaqruf Ko-Fi | [![](https://dcbadge.vercel.app/api/shield/850007095775723532?style=flat)](https://lookup.guru/850007095775723532) [![ko-fi](https://img.shields.io/badge/Support%20me%20on%20Ko--fi-F16061?logo=ko-fi&logoColor=white&style=flat)](https://ko-fi.com/linaqruf)
| Linaqruf Saweria |<a href="https://saweria.co/linaqruf"><img alt="Saweria" src="https://img.shields.io/badge/Saweria-7B3F00?style=flat&logo=ko-fi&logoColor=white"/></a>

