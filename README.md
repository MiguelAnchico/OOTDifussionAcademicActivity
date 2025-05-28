# OOTDiffusion Setup Guide

Este repositorio contiene la guía completa para instalar y configurar **OOTDiffusion** para virtual try-on, así como otros modelos GAN para preprocesamiento de imágenes.

## 📋 Tabla de Contenidos

- [Hardware Utilizado](#hardware-utilizado-en-las-pruebas)
- [Instalación de OOTDiffusion](#instalación-de-ootdiffusion)
- [Descarga de Modelos](#descarga-de-modelos)
- [Configuración del Entorno](#configuración-del-entorno)
- [Uso Básico](#uso-básico)
- [Solución de Problemas](#solución-de-problemas)
- [Transferencia de Archivos](#transferencia-de-archivos)
- [Otros Modelos GAN](#otros-modelos-gan)

## 🖥️ Hardware Utilizado en las Pruebas

- **Entorno**: Cloud GPU con 80GB de almacenamiento
- **OS**: Linux
- **CUDA**: Disponible y compatible
- **Python**: 3.10

## 🚀 Instalación de OOTDiffusion

### Paso 1: Configuración inicial del entorno

```bash
# Verificar que tenemos CUDA disponible
nvidia-smi

# Actualizar el sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias del sistema
sudo apt install -y git git-lfs build-essential cmake pkg-config libgl1-mesa-glx
```

### Paso 2: Clonar y configurar OOTDiffusion

```bash
# Clonar el repositorio
git clone https://github.com/levihsu/OOTDiffusion.git
cd OOTDiffusion

# Crear y activar entorno conda
conda create -n ootd python==3.10 -y
conda activate ootd

# Instalar PyTorch y dependencias específicas
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Instalar requirements
pip install -r requirements.txt
```

### Paso 3: Verificar instalación inicial

```bash
# Verificar que todo está instalado correctamente
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📦 Descarga de Modelos

**Fuente de los modelos**: [levihsu/OOTDiffusion en Hugging Face](https://huggingface.co/levihsu/OOTDiffusion)

### Preparar estructura de directorios

```bash
# Instalar git-lfs
git lfs install

# Crear estructura de directorios
mkdir -p checkpoints/{ootd,humanparsing,openpose}
```

### Descargar modelos específicos

```bash
# Modelos de human parsing
wget -O checkpoints/humanparsing/parsing_atr.onnx https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_atr.onnx
wget -O checkpoints/humanparsing/parsing_lip.onnx https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_lip.onnx
wget -O checkpoints/humanparsing/exp-schp-201908261155-lip.pth https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/exp-schp-201908261155-lip.pth

# Modelos principales de OOTD (pipelines completos)
git clone https://huggingface.co/levihsu/OOTDiffusion temp_ootd
cp -r temp_ootd/checkpoints/ootd checkpoints/
cp -r temp_ootd/checkpoints/openpose checkpoints/
rm -rf temp_ootd
```

### Archivos de modelos descargados

#### Human Parsing Models:

- `parsing_atr.onnx`: Modelo ONNX para human parsing ATR
- `parsing_lip.onnx`: Modelo ONNX para human parsing LIP
- `exp-schp-201908261155-lip.pth`: Modelo PyTorch para SCHP

#### OOTDiffusion Models:

- `checkpoints/ootd/`: Pipeline completo del modelo principal
- `checkpoints/openpose/`: Modelos para detección de poses

## ⚙️ Configuración del Entorno

### Solución de incompatibilidades de versiones

Si encuentras el error de importación con `huggingface_hub`, ejecuta:

```bash
# Asegúrate de estar en el entorno correcto
conda activate ootd

# Instalar versión compatible
pip install huggingface_hub==0.19.4
pip install basicsr
```

### Verificar instalaciones

```bash
# Verificar versiones instaladas
pip list | grep -E "(huggingface|diffusers|transformers)"

# Verificar que las importaciones funcionen
python -c "from huggingface_hub import hf_hub_download; print('huggingface_hub OK')"
python -c "import diffusers; print('diffusers OK')"
```

## 🎯 Uso Básico

### Preparar imágenes de prueba

```bash
# Crear estructura para imágenes de prueba
mkdir -p img_test/{clothe,person}

# Estructura esperada:
# img_test/
# ├── clothe/
# │   ├── 01260_00.jpg
# │   └── 01430_00.jpg
# └── person/
#     ├── 00891_00.jpg
#     └── 03615_00.jpg
```

### Ejecutar OOTDiffusion

```bash
# Ir al directorio de ejecución
cd OOTDiffusion/run

# Crear directorio de salida
mkdir -p images_output

# Half-body model (modelo de medio cuerpo)
python run_ootd.py \
    --model_path "../img_test/person/00891_00.jpg" \
    --cloth_path "../img_test/clothe/01260_00.jpg" \

```
