import os
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import trimesh
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline


parser = argparse.ArgumentParser()
parser.add_argument('--mesh_path', type=str, required=True)
parser.add_argument('--image_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--output_name', type=str, required=True)
args = parser.parse_args()

# 1. Load Pipeline
pipeline = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
pipeline.cuda()

# 2. Load Mesh, image & Run
mesh = trimesh.load(args.mesh_path)
mesh = mesh.geometry['GLTF']
image = Image.open(args.image_path)
output = pipeline.run(mesh, image)

# 3. Render Mesh
output.export(os.path.join(args.output_path, args.output_name + ".glb"), extension_webp=True)
