import threading
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from model import Model

import cv2


class MainApp(MDApp):
    """Main App class"""

    def run_async_app(self) -> None:
        self.model.run()

    def build(self) -> MDBoxLayout:
        self.layout = FloatLayout()  # Using FloatLayout

        # Image widget
        self.image = Image(size_hint=(1, 1))
        self.layout.add_widget(self.image)  # Add the image to the layout

        # Show render
        # 0: Show original frame
        # 1: Show frame with models
        # 2: Show depth map frame
        self.show_render = 0

        self.show_button = MDRaisedButton(
            text="Change view",  # Set the text of the button
            pos_hint={'center_x': 0.5, 'center_y': 0.1},  # Center the button
            size_hint=(None, None),
            md_bg_color=(0, 0, 0, 1),  # Set background color to black (RGBA: 0, 0, 0, 1)
            text_color=(1, 1, 1, 1)  # Set text color to white (RGBA: 1, 1, 1, 1)
        )

        self.layout.add_widget(self.show_button)  # Add the button to the layout
        self.show_button.bind(on_release=self.show_rendered)

        # self.frame_queue = Queue()  # Queue to hold frames for display
        # self.video_thread = threading.Thread(target=self.video_capture)
        # self.video_thread.start()

        # Camera capture
        # self.capture = cv2.VideoCapture(0)
        self.model = Model()
        self.model.load_yolo()
        self.model.load_midas()

        threading.Thread(target=self.run_async_app).start()
        Clock.schedule_interval(self.load_video, 1.0 / 30.0)  # Adjust frame rate if needed
        print("App started")

        return self.layout

    # def video_capture(self) -> None:
    #     """Capture video from camera"""
    #     cap = cv2.VideoCapture('./video.mp4')
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         if not self.frame_queue.full():
    #             self.frame_queue.put(frame)

    def show_rendered(self, *args) -> None:
        """Change the image rendered"""
        if self.show_render == 0:
            self.show_render = 1
        elif self.show_render == 1:
            self.show_render = 2
        elif self.show_render == 2:
            self.show_render = 0

    def load_video(self, *args) -> None:
        """Load video"""
        try:
            if self.show_render == 0:
                # success, self.frame = self.model.capture.read()
                self.frame = self.model.frame_no_render
            elif self.show_render == 1:
                self.frame = self.model.frame_rendered
            elif self.show_render == 2:
                self.frame = self.model.depth_map
            # self.frame = cv2.resize(self.frame, (1280, 720))
            buf = cv2.flip(self.frame, 0).tostring()
            texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

        except:
            pass


MainApp().run()