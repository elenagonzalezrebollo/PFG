import math
import os
os.environ['KMP_DUPLICATE_ LIB_OK']='True'
from ultralytics import YOLO
import numpy as np
import cv2 as cv2
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import torch
from gtts import gTTS
from playsound import playsound
from time import sleep
import threading

# Initialize a lock
lock = threading.Lock()
is_running = False  # Boolean flag to indicate if the speech is currently being played

# Benchmarking
from timeit import default_timer as timer

class Model:
    """ Model class

    Attributes
    ----------
    track_history : dict
        A dictionary to store the tracking history of the objects.
    track_depth_history : dict
        A dictionary to store the depth history of the objects.
    track_depth_mov_vector : dict
        A dictionary to store the depth movement vector of the objects.
    track_fps : list
        A list to store the FPS of the application.
    device : str
        The device to use for the application.
    show_depth_frame : bool
        A boolean to show the depth estimation frame.
    depth_estimation_skip : int
        An integer to skip the depth estimation.
    movement_vector_skip : int
        An integer to calculate the 2D movement vector.
    depth_average_old_frames : int
        An integer to average the depth old frames.
    depth_average_new_frames : int
        An integer to average the depth new frames.
    depth_results_to_average : int
        An integer to average the depth results.
    max_distance : int
        An integer to enable the alert.
    draw_tunnel : bool
        A boolean to draw the tunnel.
    draw_movement_vector : bool
        A boolean to draw the movement vector.
    small_offset : int
        An integer to set the small offset.
    large_offset : int
        An integer to set the large offset.
    tunnel_limits : tuple
        A tuple to save the limits of the tunnel.
    tunnel_limits_small : tuple
        A tuple to save the limits of the small tunnel.
    tunnel_left_top : tuple
        A tuple to save the top left of the tunnel.
    tunnel_left_bottom : tuple
        A tuple to save the bottom left of the tunnel.
    tunnel_right_top : tuple
        A tuple to save the top right of the tunnel.
    tunnel_right_bottom : tuple
        A tuple to save the bottom right of the tunnel.
    tunnel_left_bottom_small : tuple
        A tuple to save the bottom left of the small tunnel.
    tunnel_right_bottom_small : tuple
        A tuple to save the bottom right of the small tunnel.
    midas : torch.hub
        The MiDaS model.
    model_type : str
        The model type.
    transform : torch.hub
        The transformation.
    capture : cv2.VideoCapture
        The video capture.
    frame : np.ndarray
        The frame.
    frame_no_render : np.ndarray
        The frame without render.
    frame_rendered : np.ndarray
        The frame rendered.
    """

    def __init__(self) -> None:
        """ Initialize the application """
        self.track_history = defaultdict(lambda: [])
        self.track_depth_history = defaultdict(lambda: [])
        self.track_depth_mov_vector = defaultdict(lambda: [])
        self.track_fps = []

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {self.device}')

    def setup_settings(self, w, h) -> None:
        """ Set up the settings of the application

        Parameters
        ----------
        w : int
            The width of the frame.
        h : int
            The height of the frame.
        """
        ### Settigns ###
        # Show the depth estimation frame
        self.show_depth_frame = True

        # Number of frames to skip the depth estimation
        self.depth_estimation_skip = 4 # Higher number means more performance but less accuracy

        # Number of frames between two frames to calculate the 2D movement vector
        self.movement_vector_skip = 20

        # Number of frames to average the depth
        self.depth_average_old_frames = 6
        self.depth_average_new_frames = 6

        # Number of depth results to average
        self.depth_results_to_average = 10

        # Distance to the object to enable the alert
        self.max_distance = 12

        self.max_distance_quadrant = 3

        ## Draw options ##
        # Draw yolo items
        self.draw_yolo = True

        # Draw distance on yolo items
        self.draw_distance_yolo = True

        # Draw the tunnel ?
        self.draw_tunnel = True

        # Draw quadrants ?
        self.draw_quadrants = True

        # Draw arrows on the movement vector ?
        self.draw_movement_vector = True

        # Tunnel things
        self.small_offset = 50
        self.large_offset = 200

        # Save the limits of the tunnel to avoid recalculating things outside the tunnel
        self.tunnel_limits = (w // 2 - self.large_offset, w // 2 + self.large_offset)
        self.tunnel_limits_small = (w // 2 - self.small_offset, w // 2 + self.small_offset)

        # Save the points of the top and bottom of the tunnel for future collision calculations
        self.tunnel_left_top = (self.tunnel_limits[0], h)
        self.tunnel_left_bottom = (self.tunnel_limits[0], 0)
        self.tunnel_right_top = (self.tunnel_limits[1], h)
        self.tunnel_right_bottom = (self.tunnel_limits[1], 0)

        # Save the points of the top and bottom of the small tunnel for future collision calculations
        self.tunnel_left_top_small = (self.tunnel_limits_small[0], h)
        self.tunnel_left_bottom_small = (self.tunnel_limits_small[0], 0)
        self.tunnel_right_top_small = (self.tunnel_limits_small[1], h)
        self.tunnel_right_bottom_small = (self.tunnel_limits_small[1], 0)

        self.bottom_quadrants_ids = ['A', 'B', 'C']

    def load_midas(self) -> None:
        """ Load the MiDaS model """
        # Download the MiDaS
        self.model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # self.model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # self.model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device)
        self.midas.eval()

        # Input transformation pipeline
        transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    def load_yolo(self) -> None:
        """ Load the YOLO model """
        self.model = YOLO('./models/yolov8x.pt').to(self.device)
        self.names = self.model.model.names

    def draw_tunnels(self, frame, width, height, small_offset = 50, large_offset = 200) -> None:
        """
        Draws two tunnels on the given frame.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw the tunnels on.
        width : int
            The width of the frame.
        height : int
            The height of the frame.
        small_offset : int, optional
            The offset of the small tunnel. Default is 50.
        large_offset : int, optional
            The offset of the large tunnel. Default is 200.
        """
        # Calculate the center position
        center_width = width // 2

        # Draw the small tunnel lines
        cv2.line(frame, (center_width - small_offset, 0), (center_width - small_offset, height), (0, 0, 0), thickness=2, lineType=cv2.LINE_4)
        cv2.line(frame, (center_width + small_offset, 0), (center_width + small_offset, height), (0, 0, 0), thickness=2, lineType=cv2.LINE_4)

        # Draw the large tunnel lines
        cv2.line(frame, (center_width - large_offset, 0), (center_width - large_offset, height), (0, 0, 0), thickness=2)
        cv2.line(frame, (center_width + large_offset, 0), (center_width + large_offset, height), (0, 0, 0), thickness=2)

    def reduce_box_area_by_half(self, box) -> tuple:
        """
        Reduces the area of a given box by half, maintaining the center position.

        Parameters
        ----------
        box : tuple
            A tuple representing the (x1, y1, x2, y2) coordinates of the box.

        Returns
        -------
        new_box : tuple
            A tuple representing the new (x1, y1, x2, y2) coordinates of the reduced box.
        """
        # Extract original box coordinates and ensure they are integers
        x1, y1, x2, y2 = map(int, box[:4])

        # Calculate the width and height of the original box
        width = x2 - x1
        height = y2 - y1

        # Calculate new dimensions to reduce the area by half, using square root of 0.5
        new_width = width * math.sqrt(0.5)
        new_height = height * math.sqrt(0.5)

        # Determine the center of the original box
        center_x = x1 + width / 2
        center_y = y1 + height / 2

        # Compute new box coordinates based on the new dimensions
        new_x1 = int(center_x - new_width / 2)
        new_y1 = int(center_y - new_height / 2)
        new_x2 = int(center_x + new_width / 2)
        new_y2 = int(center_y + new_height / 2)

        # Return the new box coordinates, effectively reducing its area by half
        new_box = (new_x1, new_y1, new_x2, new_y2)
        return new_box

    def line_intersection(self, p1, p2, p3, p4) -> bool:
        """
        Determines whether extending the line from p1 to p2 infinitely intersects with the line segment from p3 to p4.

        Parameters
        ----------
        p1 : tuple
            The first point of the first line.
        p2 : tuple
            The second point of the first line.
        p3 : tuple
            The first point of the second line.
        p4 : tuple
            The second point of the second line.

        Returns
        -------
        intersection : bool
            True if the lines intersect, False otherwise.
        """
        # Calculate the slopes and intercepts of the lines (m = slope, b = y-intercept)
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p4[0] - p3[0]
        dy2 = p4[1] - p3[1]

        # Avoid division by zero for vertical lines
        if dx1 == 0 or dx2 == 0:
            return False  # Assumes vertical lines do not intersect for simplicity

        m1 = dy1 / dx1          # Slope of the first line
        b1 = p1[1] - m1 * p1[0] # y-intercept of the first line

        m2 = dy2 / dx2          # Slope of the second line
        b2 = p3[1] - m2 * p3[0] # y-intercept of the second line

        # If the slopes are the same, the lines are parallel and do not intersect
        if m1 == m2:
            return False

        # Find the intersection point of the infinitely extended lines
        x_intersect = (b2 - b1) / (m1 - m2)
        y_intersect = m1 * x_intersect + b1

        # Check if the intersection point is within the bounds of the line segment p3-p4
        if min(p3[0], p4[0]) <= x_intersect <= max(p3[0], p4[0]) and min(p3[1], p4[1]) <= y_intersect <= max(p3[1], p4[1]):
            return True

        return False

    def play_text_to_speed(self, text) -> None:
        """
        Plays the given text as speech at a specific speed using the pyttsx3 library.

        Parameters
        ----------
        text : str
            The text to be spoken.
        """
        global is_running

        if not lock.acquire(blocking=False):
            # If the lock is not available, return immediately
            return

        try:
            if is_running:
                # If another thread is already running the function, just return
                return

            is_running = True
            # Create an instance of the TTS class
            tts = gTTS(text=text, lang='en', slow=False)

            # Save the speech to a temporary file
            temp_file = os.path.dirname(__file__) + '\\audio\\temp.mp3'
            tts.save(temp_file)

            sleep(0.015)
            # Play the generated speech
            playsound(temp_file)
            # Remove the temporary file
            os.remove(temp_file)

            # sleep the thread for 1 second
            sleep(4)
        finally:
            is_running = False
            # Ensure the lock is released even if an error occurs
            lock.release()

    def thread_speech(self, text) -> None:
        # Start a new thread to play the speech
        threading.Thread(target=self.play_text_to_speed, args=(text,)).start()

    def define_grid(self, top_left, top_right, bottom_left, bottom_right) -> list:
        """ Draw a grid on the frame

        Parameters
        ----------
        top_left : tuple
            The top left point of the grid.
        top_right : tuple
            The top right point of the grid.
        bottom_left : tuple
            The bottom left point of the grid.
        bottom_right : tuple
            The bottom right point of the grid.

        Returns
        -------
        grid_pixels : list
            The grid coordinates.
        """
        # Calculate the increments for x and y coordinates
        x_increment = (top_right[0] - top_left[0]) / 3
        y_increment = (bottom_left[1] - top_left[1]) / 5

        # Initialize a list to store the grid coordinates
        grid_pixels = []

        # Loop through each row and column to generate grid coordinates
        for i in range(6):
            row = []
            for j in range(4):
                x = top_left[0] + j * x_increment
                y = top_left[1] + i * y_increment
                row.append((x, y))
            grid_pixels.append(row)

        return grid_pixels

    def run(self) -> None:
        """ Run the application """
        video_path = './video/video.mp4'
        self.capture = cv2.VideoCapture(video_path)
        assert self.capture.isOpened(), "Error al abrir la cámara"

        w, h, fps = (int(self.capture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


        self.setup_settings(w, h)

        # Start frame counter
        frame_counter = 0

        ids = [chr(i) for i in range(65, 65 + 15)]

        while self.capture.isOpened():
            # Initialize timer
            start = timer()

            success, self.frame = self.capture.read()
            try:
                self.frame_no_render = self.frame.copy()
            except:
                print("Error al copiar el frame o fin del video.")
                break
            if success:

                frame_counter += 1
                # Transform input for midas
                img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                results = self.model.track(self.frame, persist=True, verbose=False)
                boxes = results[0].boxes.xyxy.cpu()

                grid = self.define_grid(self.tunnel_left_top, self.tunnel_right_top, self.tunnel_left_bottom, self.tunnel_right_bottom)

                quadrants = self.extract_quadrants(grid)

                if self.draw_tunnel:
                    self.draw_tunnels(self.frame, w, h)

                if self.draw_quadrants:
                    for row in grid:
                        for point in row:
                            cv2.circle(self.frame, (int(point[0]), int(point[1])), 5, (0, 0, 200), -1)

                imgbatch = self.transform(img).to(self.device)

                if frame_counter % self.depth_estimation_skip == 0 or frame_counter == 1:
                    # Make a prediction
                    with torch.no_grad():
                        prediction = self.midas(imgbatch)
                        prediction = torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size = img.shape[:2],
                            mode='bicubic',
                            align_corners=False
                        ).squeeze()

                        if self.show_depth_frame:
                            output = prediction.cpu().numpy()

                            clipped_output = np.clip(output, a_min=0, a_max=255)
                            # Normalize the output to [0, 255]
                            output_normalized = cv2.normalize(clipped_output, None, 0, 255, cv2.NORM_MINMAX)

                            # Convert to unsigned 8-bit integer
                            output_uint8 = output_normalized.astype(np.uint8)

                            # Apply a colormap (COLORMAP_JET as an example, you can experiment with others)
                            colorized_output = cv2.applyColorMap(output_uint8, cv2.COLORMAP_INFERNO)

                            self.depth_map = colorized_output
                            # cv2.imshow('Depth Frame', colorized_output)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                if results[0].boxes.id is not None:
                    clss = results[0].boxes.cls.cpu().tolist()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    annotator = Annotator(self.frame, line_width=2)

                    prediction_flipped = self.flip_depth(prediction)

                    for box, cls, track_id in zip(boxes, clss, track_ids):
                        annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)]+" "+str(track_id))

                        #### Calculate the distance to the object
                        reduced_box = self.reduce_box_area_by_half(self.reduce_box_area_by_half(box))
                        x1, y1, x2, y2 = map(int, reduced_box[:4])

                        depth_values_flipped = prediction_flipped[y1:y2, x1:x2]
                        average_depth_flipped = torch.mean(depth_values_flipped.float())

                        if self.draw_distance_yolo:
                            text = f'{self.depth_to_distance(average_depth_flipped):.2f}m'

                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                            text_x = x1 + (x2 - x1 - text_size[0]) // 2
                            text_y = y1 + text_size[1] + 5

                            color = colors(int(cls), True)
                            cv2.rectangle(self.frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), color, -1)
                            cv2.putText(self.frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

                        ### Store tracking history
                        track = self.track_history[track_id]
                        track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                        if len(track) > 20:
                            track.pop(0)

                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))

                        # cv2.polylines(self.frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

                        track_depth = self.track_depth_history[track_id]
                        track_depth.append(round(float(average_depth_flipped), 2))

                        if len(track_depth) > 20:
                            track_depth.pop(0)

                        # Calculate the center X coordinate as the average of the left and right X coordinates
                        center_x = int((box[0] + box[2]) / 2)

                        if center_x > self.tunnel_limits[0] and center_x < self.tunnel_limits[1]:

                            self.process_item(box, cls, track_id, average_depth_flipped, track, track_depth)

                # Check the quadrants for objects not detected by the YOLO model
                self.check_quadrants(ids, quadrants, prediction_flipped, track)

                end = timer()
                fps = 1/(end-start)
                self.track_fps.append(fps)
                fps_text = f'FPS: {round(fps,2)}'
                cv2.putText(self.frame, fps_text, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                self.frame_rendered = self.frame

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.capture.release()
        cv2.destroyAllWindows()

        print(f'Average FPS: {sum(self.track_fps)/len(self.track_fps)}')
        print(f'Minimum FPS: {min(self.track_fps)}')
        print(f'Maximum FPS: {max(self.track_fps)}')

    def depth_to_distance(self, average_depth_flipped) -> float:
        """ Calculate the distance to the object

        Parameters
        ----------
        average_depth_flipped : float
            The average depth of the object.

        Returns
        -------
        distance : float
            The distance to the object.
        """
        if self.model_type == "DPT_Large":
            distance = average_depth_flipped/2.2
        elif self.model_type == "MiDaS_small":
            distance = average_depth_flipped/55
        return distance

    def flip_depth(self, prediction) -> torch.Tensor:
        """ Flip the depth prediction

        Parameters
        ----------
        prediction : torch.Tensor
            The depth prediction.

        Returns
        -------
        prediction_flipped : torch.Tensor
            The flipped depth prediction.
        """
        # min_val, max_val = prediction.min(), prediction.max()
        # prediction_flipped = max_val + min_val - prediction
        max_val = prediction.max()

        prediction_flipped = max_val - prediction
        return prediction_flipped

    def calc_initial_depth(self, track_depth) -> float:
        """ Calculate the initial depth of the object

        Parameters
        ----------
        track_depth : list
            The depth history of the object.

        Returns
        -------
        avg_initial_depth : float
            The average initial depth of the object.
        """

        initial_depth_array = []

        for e in range(self.depth_average_old_frames):
            initial_depth_array.append(track_depth[-self.movement_vector_skip + e])

        avg_initial_depth = sum(initial_depth_array) / len(initial_depth_array)
        return avg_initial_depth

    def calc_final_depth(self, track_depth) -> float:
        """ Calculate the final depth of the object

        Parameters
        ----------
        track_depth : list
            The depth history of the object.

        Returns
        -------
        avg_final_depth : float
            The average final depth of the object.
        """
        # Store the x values of the track_depth[-1]
        final_depth_array = []

        for e in range(self.depth_average_new_frames):
            final_depth_array.append(track_depth[-(e+1)])

        avg_final_depth = sum(final_depth_array) / len(final_depth_array)
        return avg_final_depth

    def check_quadrants(self, ids, quadrants, prediction_flipped, track) -> None:
        """ Check the quadrants for objects not detected by the YOLO model

        Parameters
        ----------
        ids : list
            The list of IDs.
        quadrants : list
            The list of quadrants.
        prediction_flipped : torch.Tensor
            The flipped depth prediction.
        track : list
            The tracking history of the object.
        """
        for i, quadrant in enumerate(quadrants):
                    # Your code for each quadrant goes here
            x1 = int(quadrant[0][0])
            x2 = int(quadrant[1][0])
            y1 = int(quadrant[2][1])
            y2 = int(quadrant[0][1])

            if ids[i] in self.bottom_quadrants_ids:
                continue

            depth_values_flipped = prediction_flipped[y1:y2, x1:x2]
            average_depth_flipped = torch.min(depth_values_flipped.float())

            if self.draw_quadrants:
                text = f'{self.depth_to_distance(average_depth_flipped):.2f}m'

                text_size = cv2.getTextSize(f'{text}, {ids[i]}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = x1 + (x2 - x1 - text_size[0]) // 2
                text_y = (y1 + y2) // 2

                if self.max_distance_quadrant < self.depth_to_distance(average_depth_flipped):
                    color = (255, 255, 255)
                else:
                    color = (0, 0, 255)
                cv2.rectangle(self.frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), color, -1)
                cv2.putText(self.frame, f'{text}, {ids[i]}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            track_depth = self.track_depth_history[ids[i]]
            track_depth.append(round(float(average_depth_flipped), 2))

            if len(track_depth) > 20:
                track_depth.pop(0)

            self.process_quadrant(ids[i], average_depth_flipped, track, track_depth)

    def extract_quadrants(self, grid) -> list:
        """ Extract the quadrants from the grid

        Parameters
        ----------
        grid : list
            The grid.

        Returns
        -------
        quadrants : list
            The quadrants.
        """
        quadrants = [grid[i][j:j+2] + grid[i+1][j:j+2] for i in range(0, 5) for j in range(0, 3)]

        return quadrants

    def process_quadrant(self, track_id, average_depth_flipped, track, track_depth) -> None:
        """ Process the quadrant

        This function processes the quadrant and calculate the depth movement vector.

        Parameters
        ----------
        track_id : str
            The track ID of the object.
        average_depth_flipped : float
            The average depth of the object.
        track : list
            The tracking history of the object.
        track_depth : list
            The depth history of the object.
        """
        if len(track) > self.movement_vector_skip-1:
            avg_initial_depth = self.calc_initial_depth(track_depth)

            avg_final_depth = self.calc_final_depth(track_depth)

            self.track_depth_mov_vector[track_id].append(avg_final_depth - avg_initial_depth)

            avg_depth_mov_vector = sum(self.track_depth_mov_vector[track_id][-self.depth_results_to_average:]) / self.depth_results_to_average

            self.alert_if_collision_quadrant(average_depth_flipped, avg_depth_mov_vector)

    def alert_if_collision_quadrant(self, average_depth_flipped, avg_depth_mov_vector) -> None:
        """ Alert if there is a collision in the quadrant

        This function checks if there is an object moving towards the camera and is close enough to trigger an alert.

        Parameters
        ----------
        average_depth_flipped : float
            The average depth of the object.
        avg_depth_mov_vector : float
            The average depth movement vector of the object.
        """
        if avg_depth_mov_vector < -2:
            distance = self.depth_to_distance(average_depth_flipped)
            if distance < self.max_distance_quadrant:
                self.thread_speech("Alert: Unrecognized object is near you")
                # pass

    def process_item(self, box, cls, track_id, average_depth_flipped, track, track_depth) -> None:
        """ Process the item

        This function processes the item and calculate the 2D and 3D movement vectors.

        Parameters
        ----------
        box : tuple
            The box coordinates of the object.
        cls : int
            The class of the object.
        track_id : str
            The track ID of the object.
        average_depth_flipped : float
            The average depth of the object.
        track : list
            The tracking history of the object.
        track_depth : list
            The depth history of the object.
        """
        if len(track) > self.movement_vector_skip-1:
            initial_point = track[-self.movement_vector_skip]

            final_point = track[-1]

            if self.draw_movement_vector:
                # Draw the movement vector on the frame
                cv2.arrowedLine(self.frame, initial_point, final_point, (175, 0, 178), 4)

            ### Calculate the 3D movement vector
            # Use the average_depth_flipped value to calculate the 3D movement vector
            # The average_depth_flipped value is the average depth of the object in the box

            avg_initial_depth = self.calc_initial_depth(track_depth)

            avg_final_depth = self.calc_final_depth(track_depth)

            # Store the 3D initial and final points
            initial_point_3d = (initial_point[0], initial_point[1], avg_initial_depth)
            final_point_3d = (final_point[0], final_point[1], avg_final_depth)

            # Store the 3D movement vector
            movement_vector_3d = (
                final_point_3d[0] - initial_point_3d[0], # X
                final_point_3d[1] - initial_point_3d[1], # Y
                final_point_3d[2] - initial_point_3d[2]  # Z
            )

            # Store the depth movement vector
            self.track_depth_mov_vector[track_id].append(movement_vector_3d[2])

            # Store the average of the last x values of the track_depth_mov_vector
            avg_depth_mov_vector = sum(self.track_depth_mov_vector[track_id][-self.depth_results_to_average:]) / self.depth_results_to_average

            self.alert_if_collision_item(box, cls, average_depth_flipped, initial_point, final_point, avg_depth_mov_vector)

    def alert_if_collision_item(self, box, cls, average_depth_flipped, initial_point, final_point, avg_depth_mov_vector) -> None:
        """ Alert if there is a collision with the object

        This function checks if there is an object moving towards the tunnel and is close enough to trigger an alert.

        Parameters
        ----------
        box : tuple
            The box coordinates of the object.
        cls : int
            The class of the object.
        average_depth_flipped : float
            The average depth of the object.
        initial_point : tuple
            The initial point of the object.
        final_point : tuple
            The final point of the object.
        avg_depth_mov_vector : float
            The average depth movement vector of the object.
        """
        # Initialize variables to store the status of the object
        aiming_tunnel_status = False # True if the object is aiming the tunnel
        depth_status = False # True if the object is moving towards
        distance_status = False # True if the object is close than x meters

        top_line_intersection = False # True if the object points to the top line of the tunnel
        bottom_line_intersection = False # True if the object points to the bottom line of the tunnel

        # top_line_intersection = self.line_intersection(initial_point, final_point, self.tunnel_left_top_small, self.tunnel_right_top_small)
        bottom_line_intersection = self.line_intersection(initial_point, final_point, self.tunnel_left_bottom_small, self.tunnel_right_bottom_small)

        # if angle > 70 and angle < 110 or angle > -110 and angle < -70:
        if top_line_intersection or bottom_line_intersection:
            aiming_tunnel_status = True

        if avg_depth_mov_vector < -2:
            depth_status = True

        distance = self.depth_to_distance(average_depth_flipped)
        if distance < self.max_distance:
            distance_status = True

        # If all the conditions are met, draw a red rectangle around the object
        if aiming_tunnel_status and depth_status and distance_status:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 0), 5)

            # Draw the alert message on the frame
            alert_text = f'Alert: {self.names[int(cls)]} is near you'
            print(alert_text)
            self.thread_speech(alert_text)
