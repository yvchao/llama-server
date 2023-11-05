#!/bin/bash

# Check if bash and conda are installed
if ! command -v bash &> /dev/null || ! command -v conda &> /dev/null; then
  echo "Bash or Conda is not installed. Exiting."
  exit 1
fi

set -e  # Stop script on error

# Variables
ENV_NAME="llama-cpp"
PYTHON_VERSION="3.10"

LLAMA_CPP_VERSION="b1488"
CUDA_VERSION="11.4.0"

printf "Target CUDA version: $CUDA_VERSION.\nPlease ensure it is compatible with your GPU driver.\n"
read -p "Continue installation? [Y/n]: " choice
if [[ "$choice" == "n" || "$choice" == "N" ]]; then
  echo "Exiting script."
  exit 1
fi

# Load conda hooks
eval "$(conda shell.bash hook)"

setup_cuda_env()
{
  # Create a conda environment
  echo "Creating a conda environment named $ENV_NAME with Python $PYTHON_VERSION..."
  conda create --name "$ENV_NAME" python="$PYTHON_VERSION" --yes

  # Activate the conda environment
  echo "Activating the conda environment..."
  conda activate "$ENV_NAME"

  # Install cuda toolkit
  echo "Installing cuda toolkit $CUDA_VERSION..."
  conda install -c "nvidia/label/cuda-$CUDA_VERSION" cuda-toolkit --yes

  # Install cmake
  echo "Installing CMake..."
  conda install -c anaconda cmake --yes
}

# Check if the conda environment already exists
env_check=$(conda env list | awk '{print $1}' | grep -w "$ENV_NAME" || true)
if [ "$env_check" == "$ENV_NAME" ]; then
  read -p "Environment $ENV_NAME already exists. Do you want to reset it? [y/N]: " choice
  if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    echo "Deleting environment $ENV_NAME..."
    conda env remove --name "$ENV_NAME" --yes
    setup_cuda_env
  else
    echo "Reinstall llama.cpp in $ENV_NAME..."
    # Activate the conda environment
    echo "Activating the conda environment..."
    conda activate "$ENV_NAME"
  fi
else
  setup_cuda_env
fi

# Optional: Verify CUDA
echo "Verifying CUDA installation..."
nvcc --version

# Optional: Verify CMake
echo "Verifying CMake installation..."
cmake --version

echo "Installing llama.cpp..."
git clone --branch $LLAMA_CPP_VERSION --depth 1 https://github.com/ggerganov/llama.cpp.git

cd llama.cpp
echo "Apply fix for system prompt..."
git apply ../system_prompt.patch

echo "Start building server binary..."
cmake -B build -DLLAMA_CUBLAS=ON -DLLAMA_AVX=ON -DLLAMA_AVX2=ON
cmake --build build --config Release -j --target server # we only need the server
cd ..
cp "./llama.cpp/build/bin/server" ./llama_server
yes | rm -r llama.cpp

echo "Installation complete."
