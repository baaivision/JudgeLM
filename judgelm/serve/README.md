# JudgeLM Gradio

To serve using the web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the webserver and model workers. 

Here are the commands to follow in your terminal:

## Step 1. Start Controller

```bash
python -m judgelm.serve.controller
```

This controller manages the distributed workers.

## Step 2. Start Model Worker 

```bash
python -m judgelm.serve.model_worker --model-path [MODEL_PATH] --num-gpus [NUM_GPUS] --port [PORT]
```

Arguments:
  - `[MODEL_PATH]` is the path to the judge weights, which can be a local folder.
  - `[NUM_GPUS]` is the total number of GPU to use that 33B model needs 2 GPUS, 7B and 13B models need 1 GPU.
  - `[PORT]` is the port for the current model worker. You need to choose different port for different model.

e.g.,

```bash
python -m judgelm.serve.model_worker --model-path "/share/project/lianghuizhu/JudgeLM-Project/judgelm-13b-v1.0-full-model" --num-gpus 1 --port 21003
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller .

## Step 3. Start Gradio

```bash
python -m judgelm.serve.gradio_web_server_judgelm
```

This is the user interface that users will interact with.

By following these steps, you will be able to serve your models using the web UI. You can open your browser and chat with a model now.
If the models do not show up, try to reboot the gradio web server.