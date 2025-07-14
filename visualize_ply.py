import open3d as o3d
import numpy as np

# Load a PLY file (assuming it's a point cloud)
# If your PLY contains a mesh (vertices and faces), Open3D can also load it as a TriangleMesh
POINT_CLOUD_FILE = "demo_data/tomato_soup_can/mesh/obj_000009.ply"
MESH_FILE = "demo_data/tomato_soup_can/mesh/converted_obj_000009.obj"
FRAME_SIZE = 50

MESH = False

if not MESH:
    pcd = o3d.io.read_point_cloud(POINT_CLOUD_FILE)
else:
    try:
        mesh = o3d.io.read_triangle_mesh(MESH_FILE)
        # If it's a mesh, you might want to visualize it as a mesh or convert to point cloud
        if mesh.has_vertices():
            pcd = mesh.sample_points_uniformly(number_of_points=10000) # Sample points from mesh
        else:
            print("The PLY file doesn't seem to contain vertices for point cloud or mesh.")
            exit()
    except Exception as e_mesh:
        print(f"Could not read as point cloud or mesh: {e_mesh}")
        exit()


if pcd.is_empty():
    print("Point cloud is empty. Check your PLY file path and content.")
else:
    print(f"Loaded point cloud with {len(pcd.points)} points.")
    if not MESH:
        geometries_to_visualize = [pcd]
        global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=FRAME_SIZE, origin=[0, 0, 0])
        geometries_to_visualize.append(global_frame)

        o3d.visualization.draw_geometries(geometries_to_visualize,
                                        window_name="PLY Point Cloud Viewer",
                                        width=800,
                                        height=600,
                                        left=50,
                                        top=50,
                                        point_show_normal=False, # Set to True to see normals if present
                                        mesh_show_wireframe=False) # Only relevant if visualizing a mesh
    else:
        # If your PLY file is a mesh, you'd visualize it like this:
        mesh = o3d.io.read_triangle_mesh(MESH_FILE)
        o3d.visualization.draw_geometries([mesh], window_name="PLY Mesh Viewer")