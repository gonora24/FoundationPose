from hardware.hardware_orbbec import ORBBEC
import time
import cv2
import argparse

def save_camera_image(rgb_path):

    devices = ORBBEC.get_devices(1, 360, 640)
    device = devices[0]
    assert device.connect(), f"Connection to {device.name} failed"
    
    MAX_ATTEMPTS = 20

    print("Warming up camera...")
    for _ in range(MAX_ATTEMPTS):
        data = device.get_sensors()
        if data and data.get("rgb") is not None:
            print("RGB stream is live.")
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("Failed to receive RGB frame after warm-up.")

    data = device.get_sensors()
    color = data['rgb']
    cv2.imwrite(rgb_path, color)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_path', type=str)
    args = parser.parse_args()
    save_camera_image(args.rgb_path)