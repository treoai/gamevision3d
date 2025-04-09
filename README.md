# GameVision3D

Codebase for Treo AI GameVision3D for **SCAI AI League**.

This code demonstrates functionalities for player & ball tracking as well as field line callibration.

## Installation

Follow these steps to set up the environment:

1. **Create and activate a new Python environment**  
   (using `venv`, `conda`, or your preferred tool)

2. **Install the latest version of PyTorch with CUDA support**  
   Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for the appropriate command based on your system.  
   Example (with pip and CUDA 11.8):  
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install the main package**  
   From the root directory of the repository:
   ```bash
   pip install .
   ```

4. **Set up the soccer example**  
   Navigate to the `example/soccer` directory:
   ```bash
   cd example/soccer
   pip install -r requirements.txt
   ./setup.sh
   ```

5. **Run the GUI**  
   From the same directory:
   ```bash
   python gui.py
   ```
