import queue
import socket
import struct
import threading
import cv2
import numpy as np
from ultralytics import YOLO

SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600
WIDTH = 640
HEIGHT = 360

class VisionRX:
    def __init__(self, data, model_path="angmar_v1.pt"):
        self.frame_queue = queue.Queue(maxsize=2)
        self.data = data
        self.is_running = True
        self.window_name = "FPV Feed"
        self.model = YOLO(model_path)

        self.thread = threading.Thread(
            target=self._vision_loop, daemon=True
        )
        self.thread.start()

    def stop(self):
        self.is_running = False

    def get_thread_for_join(self):
        self.stop()
        return self.thread

    def _vision_loop(self):
        header_format = "<IHHIIQ"
        header_sz = struct.calcsize(header_format)
        frames = {}

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))

        while self.is_running:
            try:
                sock.settimeout(1.0)
                packet, addr = sock.recvfrom(65536)
            except socket.timeout:
                continue

            header = packet[:header_sz]
            payload = packet[header_sz:]

            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = struct.unpack(
                header_format, header
            )

            if frame_id not in frames:
                frames[frame_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "size": jpeg_size,
                    "time": sim_time_ns,
                }

            frames[frame_id]["chunks"][chunk_id] = payload

            if len(frames[frame_id]["chunks"]) == total_chunks:
                jpeg_bytes = bytearray()
                frame_complete = True
                for i in range(total_chunks):
                    if i not in frames[frame_id]["chunks"]:
                        frame_complete = False
                        break
                    jpeg_bytes.extend(frames[frame_id]["chunks"][i])

                if not frame_complete:
                    del frames[frame_id]
                    continue

                img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if image is not None:
                    self.process_frame(image)

                del frames[frame_id]

        sock.close()

    def process_frame(self, img):
        results = self.model.track(img, verbose=False, tracker="bytetrack.yaml")
        annotated_frame = results[0].plot()

        try:
            self.frame_queue.put_nowait(annotated_frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(annotated_frame)
            except queue.Empty:
                pass

        unsorted_gates = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_centered = (x1 + x2) / 2.0 - WIDTH / 2.0
                y_centered = (y1 + y2) / 2.0 - HEIGHT / 2.0
                depth = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                unsorted_gates.append([x_centered, y_centered, float(depth)])

        if len(unsorted_gates) > 0:
            unsorted_array = np.array(unsorted_gates)
            sort_indices = np.argsort(unsorted_array[:, -1])[::-1]
            self.data["gates"] = unsorted_array[sort_indices].tolist()
        else:
            self.data["gates"] = []

    def update_window(self, OUTPUT_MODE=False, video_writer=None):
        try:
            frame = self.frame_queue.get_nowait()
            if len(self.data.get("gates", [])) > 0:
                cv2.line(
                    frame,
                    (int(WIDTH/2), int(HEIGHT/2)),
                    (int(WIDTH/2 + self.data["gates"][0][0]), int(HEIGHT/2 + self.data["gates"][0][1])),
                    (0, 0, 255),
                    3
                )
            cv2.imshow(self.window_name, frame)
            if OUTPUT_MODE and video_writer is not None:
                video_writer.write(frame)
        except queue.Empty:
            pass

        cv2.waitKey(1)