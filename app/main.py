import flet as ft
import flet.canvas as cv
import numpy as np
import sys
from PIL import Image
from io import BytesIO
import base64
import torch
from torchvision.transforms import GaussianBlur, ToPILImage

from core.model import DigitRecognitionModel
import core.config as config

AUTHOR = "Arnau (arnaupy)"
PANEL_WIDTH = PANEL_HEIGHT = 400
PADDING = 2
PIXELS = 28
BACKGOUND_COLOR = "#ffffff"  # white
PIXELS_COLOR = "#000000"  # black
# (PIXELS + 1)*PADDING + PIXELS*RECTANGLE_WIDTH = PANEL_WIDTH ->
RECTANGLE_WIDTH = RECTANGLE_HEIGHT = (PANEL_WIDTH - (PIXELS + 1) * PADDING) / PIXELS
BLUR = 1
PREVIEW_FILENAME = "preview.png"
GITHUB_URL = "https://github.com/arnaupy/MLProjects/tree/develop/DigitRecognition"
GITHUB_ICON = "github.svg"
DATASET_NAME = "MNIST"


class Panel:
    """Panel to draw the numbers storing the canvas and an array with the cells drawn"""

    def __init__(self):
        self.data = self._get_empty_panel()
        self.canvas = cv.Canvas()
        self.fill()
        self.create_blanck_panel()

    def _get_empty_panel(self):
        """Return an array of shape (PIXELS, PIXELS) full of 0s"""
        return np.full((PIXELS, PIXELS), 0, np.uint8)

    def clear(self):
        """Clear the canvas and data"""
        self.canvas.clean()
        self.data = self._get_empty_panel()

    def update(self, row, col, opacity):
        """Update panel row and col data with an specific opacity"""
        self.data[row][col] = 255 * (1 - opacity)

    def fill(self):
        """Fill the entire canvas background in `BACKGROUND_COLOR`"""
        self.canvas.shapes.append(cv.Fill(ft.Paint(BACKGOUND_COLOR)))

    def create_blanck_panel(self):
        """Fill the canvas with black rectangles representing pixels"""
        for row in range(PIXELS):
            for col in range(PIXELS):
                x_coordinate = 0 if row == 0 else (RECTANGLE_WIDTH + PADDING) * row
                y_coordinate = 0 if col == 0 else (RECTANGLE_HEIGHT + PADDING) * col
                self.canvas.shapes.append(
                    cv.Rect(
                        x_coordinate + PADDING,
                        y_coordinate + PADDING,
                        RECTANGLE_WIDTH,
                        RECTANGLE_HEIGHT,
                        paint=ft.Paint(PIXELS_COLOR),
                    )
                )


class MouseController:
    """Takes care of managing mouse gestures"""

    def __init__(self, panel: Panel):
        self.panel = panel
        self.add_gesture_detector()

    def add_gesture_detector(self):
        """Add the mouse moving gesture to the panel canvas"""
        self.panel.canvas.content = ft.GestureDetector(
            on_pan_update=self.pan_update, drag_interval=5
        )

    def pan_update(self, e: ft.DragUpdateEvent):
        """Paints the panel canvas and store its data array"""
        col = int((e.local_x - PADDING) / (RECTANGLE_WIDTH + PADDING))
        row = int((e.local_y - PADDING) / (RECTANGLE_HEIGHT + PADDING))
        if row <= 0 or col <= 0 or row >= PIXELS or col >= PIXELS:
            return
        rectangle_index = col * PIXELS + row + 1
        self.panel.canvas.shapes[rectangle_index].paint = ft.Paint(
            color=ft.colors.with_opacity(0, "#ffffff")
        )
        self.panel.update(row, col, 0)
        for neighbour in get_neighbours(row, col):
            rectangle = self.panel.canvas.shapes[
                neighbour[1] * PIXELS + neighbour[0] + 1
            ]
            try:
                opacity = float(rectangle.paint.color.split(",")[1]) - 0.1
                if opacity < 0.0:
                    opacity = 0
            except:
                opacity = 0.5
            rectangle.paint = ft.Paint(color=ft.colors.with_opacity(opacity, "#000000"))
            self.panel.update(neighbour[0], neighbour[1], opacity)
        self.panel.canvas.update()


def get_neighbours(row: int, col: int):
    """Return the neighbour cells as a tuple of (row, col)"""
    cells = set()
    for i in range(-BLUR, BLUR + 1):
        for j in range(-BLUR, BLUR + 1):
            x = j + col
            y = i + row
            if x == col and y == row:
                continue
            else:
                if (x >= 0 and x < PIXELS - 1) and (y >= 0 and y < PIXELS - 1):
                    cells.add((y, x))
    return cells


def main(page: ft.Page):
    page.title = "Digit Recognizer"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    page.fonts = {
        "RobotoSlab": "https://github.com/google/fonts/raw/main/apache/robotoslab/RobotoSlab%5Bwght%5D.ttf"
    }

    # Model name
    if len(sys.argv) > 2:
        print("Usage: <model_name.pth>")
        page.window_destroy()
    elif len(sys.argv) == 1:
        model_name = "model.pth"

    else:
        model_name = sys.argv[1]
        if not model_name.endswith(".pth"):
            print("Usage: <model_name.pth>")
            page.window_destroy()

    def reset(e):
        panel.clear()
        panel.fill()
        panel.create_blanck_panel()
        controller.add_gesture_detector()
        panel.canvas.update()
        prediction.value = "Prediction: "
        page.update()

    def preview(e):
        img = Image.fromarray(panel.data)
        img = img.resize((PANEL_WIDTH, PANEL_HEIGHT))
        buff = BytesIO()
        img.save(buff, format="JPEG")
        image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
        preview_image.content = ft.Image(
            src_base64=image_string,
            border_radius=5,
            width=PANEL_WIDTH,
            height=PANEL_HEIGHT,
        )
        preview_image.update()
        page.update()

    def predict(e):
        x = torch.Tensor(panel.data).unsqueeze(dim=0)

        model.eval()
        with torch.inference_mode():
            y_logits = model(x)
            prediction.value = (
                f"Prediction: {torch.argmax(torch.softmax(y_logits, dim = 1))}"
            )
        page.update()

    # Load the model to predict
    model = DigitRecognitionModel(in_features=config.IMAGE_SIZE**2, out_features=10)
    parameters = torch.load(f"{config.MODELS_PATH}/{model_name}")
    model.load_state_dict(parameters)

    # Instanciate panel
    panel = Panel()

    # Instanciate panel mouse controller
    controller = MouseController(panel)

    # Instanciate game description
    description = ft.Text(
        "Draw the number you want to predict and click the Predict button. Click Reset to predict another number.",
        text_align=ft.TextAlign.JUSTIFY,
    )

    # Instanciate prediction text
    prediction = ft.Text("Prediction:", size=40)

    # Instanciate preview image and text
    preview_image = ft.Container(
        ft.Row(
            [
                ft.Text(
                    "Click Preview Button to show the augmented \n version of the drawing",
                    text_align=ft.TextAlign.CENTER,
                )
            ],
            width=PANEL_WIDTH,
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )
    preview_text = ft.Text("Preview", size=40)

    # Instanciate button style
    style = ft.ButtonStyle(color="white", bgcolor="gray")

    page.add(
        ft.Row(
            [
                ft.Column(
                    [
                        ft.Text(
                            "DigitRecognition App",
                            size=40,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        description,
                        ft.Text(
                            f"Model used: {model_name}",
                            size=15,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        ft.Text(
                            f"Training dataset: {DATASET_NAME}",
                            size=15,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        ft.Text(
                            f"Author: {AUTHOR}",
                            size=10,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        ft.Row(
                            [
                                ft.Stack(
                                    [
                                        ft.Image(
                                            src=GITHUB_ICON,
                                            width=20,
                                            height=20,
                                            color=BACKGOUND_COLOR,
                                        ),
                                        ft.FilledButton(
                                            url=GITHUB_URL,
                                            width=20,
                                            height=20,
                                            opacity=0,
                                        ),
                                    ],
                                )
                            ],
                        ),
                    ],
                    width=PANEL_WIDTH,
                ),
                ft.Column(
                    [
                        ft.Container(
                            ft.Row([prediction], alignment=ft.MainAxisAlignment.CENTER),
                            width=PANEL_WIDTH,
                        ),
                        ft.Container(
                            panel.canvas,
                            border_radius=5,
                            width=PANEL_WIDTH,
                            height=PANEL_HEIGHT,
                        ),
                        ft.Container(
                            ft.Row(
                                [
                                    ft.ElevatedButton(
                                        content=ft.Text("Reset", size=25),
                                        on_click=reset,
                                        style=style,
                                    ),
                                    ft.ElevatedButton(
                                        content=ft.Text("Predict", size=25),
                                        on_click=predict,
                                        style=style,
                                    ),
                                    ft.ElevatedButton(
                                        content=ft.Text("Preview", size=25),
                                        on_click=preview,
                                        style=style,
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                            ),
                            width=PANEL_WIDTH,
                        ),
                    ]
                ),
                ft.Column(
                    [
                        ft.Container(
                            ft.Row(
                                [preview_text],
                                alignment=ft.MainAxisAlignment.CENTER,
                                width=PANEL_WIDTH,
                            )
                        ),
                        preview_image,
                    ]
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_EVENLY,
        )
    )

    page.update()


ft.app(main)
