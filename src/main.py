import gradio as gr
import torch
import numpy as np
import open3d as o3d
import base64
import json
import datetime
import sys
import io
import trimesh
import requests

from PIL import Image
from pathlib import Path
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from azure.storage.blob import BlobServiceClient


feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', './data/input/cats.jpg')

def webhook_call(glb_url):
    glbfile = glb_url.split("/")[-1]
    userID = glbfile.split("-")[0]
    requestID = glbfile.split("-")[-1].split(".")[0]
    
    payload = {
        "userID": userID,
        "requestID": requestID,
        "glbModelUrl": glb_url
    }

    # Make the REST API call
    response = requests.post("https://ultronbackenddev0.azurewebsites.net/api/v1/ultron3dModel", json=payload)

def read_image_from_blob(blob_conn_str, blob_container_name, file_name):
    # Connect to the Blob Storage account and get the container client
    blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
    container_client = blob_service_client.get_container_client(blob_container_name)
    
    # Download the image file from the container
    blob_client = container_client.get_blob_client(file_name)
    image_data = blob_client.download_blob().content_as_bytes()
    
    # Print the size of the image file
    image_size = len(image_data)
    print(f"Downloaded image '{file_name}' with size {image_size} bytes")
    
    return image_data
    

def upload_file_to_blob(blob_conn_str, blob_container_name, local_file_path, blob_file_name):
     #container_name = "filefoldertest"
     blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
     blob_client = blob_service_client.get_blob_client(container=blob_container_name, blob=blob_file_name)
     try:
       blob_service_client.create_container(blob_container_name)
     except:
       pass

     with open(local_file_path,"rb") as data:
        blob_client.upload_blob(data, overwrite=True)
        print(f"Uploaded {blob_file_name}.")   
        
     uploaded_file_url = blob_client.url
     print(f"File URL: {uploaded_file_url}")
     
     return uploaded_file_url    


def process_image(image_path, gender, auth):
#    blob_conn_str = "DefaultEndpointsProtocol=https;AccountName=frontenddatabase;AccountKey=b3g7sqtI/1Ju5YlWDHsOCnb6kNQVTqoz8aG2Ds29FusP7KlKRnlmalbZqYFhEbhxpOKdvYNKCvxj+AStn9r/5A==;EndpointSuffix=core.windows.net"
#    blob_container_name = "api-container"
    blob_conn_str = "DefaultEndpointsProtocol=https;AccountName=metadevselfie;AccountKey=GW21WZOMCJhOk0fSjSewP8VzLum6oP85rN65k4ncIonqM5VU2FePqB3SIfva+FutasOQlaALkkLy+AStTUhBeA==;EndpointSuffix=core.windows.net"
    blob_container_name1 = "selfie"
    blob_container_name2 = "glb-test"
    file_name = image_path.split("/")[-1]
    
    print(">>>",bool(auth),">>",bool(gender))
    
    if not auth or not gender or not image_path:
        print("nonell")
        err_msg = "Please enter the Inputs!"
        json_string = "Null"
        return [err_msg,json_string]

    else:
        image_path = read_image_from_blob(blob_conn_str, blob_container_name1, file_name)
        print("file read done")
        err_msg = "ALL OK!"
        print("Read Image")
        image_raw = Image.open(io.BytesIO(image_path))
        image = image_raw.resize(
            (800, int(800 * image_raw.size[1] / image_raw.size[0])),
            Image.Resampling.LANCZOS
        )

        # prepare image for the model
        print("Prepare image for model")
        encoding = feature_extractor(image, return_tensors="pt")

        # forward pass
        with torch.no_grad():
            outputs = model(**encoding)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()
        depth_image = (output * 255 / np.max(output)).astype('uint8')

        try:
            gltf_path = create_3d_obj(np.array(image), depth_image, file_name.split(".")[0])
            file_url = upload_file_to_blob(blob_conn_str, blob_container_name2, gltf_path, auth+".glb")
            
            # img = Image.fromarray(depth_image)
            with open(gltf_path, 'rb') as f:
                glb_data = f.read()

            encoded_data = base64.b64encode(glb_data).decode('utf-8')
            now = datetime.datetime.now()
            glb_dict = {#"glb_data" : encoded_data,
                        "gender" : gender,
                        "name" : auth,
                        "glb_url": file_url,
                        "time" : now.strftime("D-%Y-%m-%d")+now.strftime("_T-%H-%M-%S")}
            glb_dict1 = {"log" : err_msg}            
            json_string = json.dumps(file_url) #glb_dict
            json_string1 = json.dumps(err_msg)
            webhook_call(file_url)
            return json_string


        except:
            print("Error reconstructing 3D model")
            raise Exception("Error reconstructing 3D model")

def create_3d_obj(rgb_image, depth_image, image_path, depth=10):
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(rgb_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d, depth_o3d, convert_rgb_to_intensity=False)
    w = int(depth_image.shape[1])
    h = int(depth_image.shape[0])

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(w, h, 500, 500, w/2, h/2)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsic)

    print('normals')
    pcd.normals = o3d.utility.Vector3dVector(
        np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 1000.]))
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    pcd.transform([[-1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    #print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh_raw, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=True)

    voxel_size = max(mesh_raw.get_max_bound() - mesh_raw.get_min_bound()) / 256
    #print(f'voxel_size = {voxel_size:e}')
    mesh = mesh_raw.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)

    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_crop = mesh.crop(bbox)
    gltf_path = f'./data/output/test.glb'
    o3d.io.write_triangle_mesh(
        gltf_path, mesh_crop, write_triangle_uvs=True)
    return gltf_path


title = "Demo: zero-shot depth estimation with DPT + 3D Point Cloud"
description = "This demo is a variation from the original <a href='https://huggingface.co/spaces/nielsr/dpt-depth-estimation' target='_blank'>DPT Demo</a>. It uses the DPT model to predict the depth of an image and then uses 3D Point Cloud to create a 3D object."
examples =[['cats.jpg']]

if __name__ == '__main__':
    iface = gr.Interface(fn=process_image,
                        inputs=[
                            gr.Text(
                                label="Input Image*",
                                ),
                            gr.Text(
                                label="Select Gender*",
                                ),
                            gr.Text(
                                label="User Name*",
                                )  
                            ],
                        outputs = gr.Text(
                                label="Output",
                                )
                        )  
    iface.launch(debug=True,server_name="0.0.0.0", server_port=8022)       