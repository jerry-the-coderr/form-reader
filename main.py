# a scaled down version of the main concept project

import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

import easyocr

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import ( QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QFormLayout, QFileDialog, QMessageBox, QPlainTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QStackedWidget, QProgressBar, QScrollArea, )
# ============================================================
# CONFIGURATION
# ============================================================

TEMPLATE_IMAGE = "form.png"
FIELDS_JSON = "fields.json"

# Field types used to constrain EasyOCR.
FIELD_TYPES = {
    "name": "text",
    "class": "num",
    "section": "text",
    "roll_number": "num",
    "phone_number": "num",
    "blood_group": "blood",
}

OUTPUT_JSON = "scan_results.json"
OUTPUT_DB = "scan_results.db"

# Below this = low confidence
LOW_CONFIDENCE = 0.65

# Below this = medium confidence
GOOD_CONFIDENCE = 0.85

OCR_LANGUAGES = ["en"]

# Important for your low-end machine
OCR_GPU = False



# ============================================================
# FIELD DEFINITIONS / TEMPLATE
# ============================================================

def load_field_definitions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_field_definitions(path, definitions):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(definitions, f, indent=4)


def save_template_image(image, path=TEMPLATE_IMAGE):
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Could not save template image: {path}")


def load_template_size(path):
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"Could not load template image: {path}")
    height, width = image.shape[:2]
    return width, height


def get_field_type(field_name):
    return FIELD_TYPES.get(field_name, "text")


def get_allowlist(field_name):
    field_type = get_field_type(field_name)

    if field_type == "num":
        return "0123456789"

    if field_type == "blood":
        return "ABO+-"

    # Keep spaces/punctuation available for names and sections.
    return None


def normalize_ocr_value(field_name, text):
    text = str(text).strip()

    if get_field_type(field_name) == "num":
        return "".join(ch for ch in text if ch.isdigit())

    if get_field_type(field_name) == "blood":
        value = text.upper().replace(" ", "")
        # Common OCR variants such as O+ are left visible for verification;
        # only whitespace is removed because confidence/verification should
        # not silently invent the blood group.
        return value

    return " ".join(text.split())


# ============================================================
# PDF -> IMAGES
# ============================================================

def image_from_pdf_page(page):

    """
    Render one PDF page into an OpenCV image.

    We render the entire page instead of trying to find
    individual embedded PDF images because the form field
    coordinates belong to the entire page.
    """

    pix = page.get_pixmap(
        dpi=200,
        alpha=False
    )

    data = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    )

    image = data.reshape(
        pix.height,
        pix.width,
        pix.n
    )

    if pix.n == 4:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2BGR
        )

    else:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    return image


def load_input(path):

    """
    Returns:

        [
            (
                source_filename,
                page_number,
                image
            ),
            ...
        ]
    """

    path = Path(path)

    suffix = path.suffix.lower()

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if suffix == ".pdf":

        result = []

        document = pymupdf.open(
            str(path)
        )

        try:

            for page_number, page in enumerate(
                document,
                start=1
            ):

                image = image_from_pdf_page(
                    page
                )

                result.append(
                    (
                        path.name,
                        page_number,
                        image
                    )
                )

        finally:

            document.close()

        return result

    # --------------------------------------------------------
    # Normal image
    # --------------------------------------------------------

    image = cv2.imread(
        str(path)
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image: {path}"
        )

    return [
        (
            path.name,
            1,
            image
        )
    ]


# ============================================================
# CROP FORM FIELDS
# ============================================================

def crop_fields(
    image,
    field_definitions,
    template_width,
    template_height
):

    """
    The coordinates in fields.json belong to the template.

    If the scanned image is a different resolution,
    automatically scale the coordinates.
    """

    image_height, image_width = image.shape[:2]

    scale_x = image_width / template_width
    scale_y = image_height / template_height

    crops = {}
    rectangles = {}

    for field_name, field in field_definitions.items():

        x = int(
            field["x"] * scale_x
        )

        y = int(
            field["y"] * scale_y
        )

        w = int(
            field["width"] * scale_x
        )

        h = int(
            field["height"] * scale_y
        )

        # Keep coordinates inside image
        x1 = max(
            0,
            min(
                x,
                image_width - 1
            )
        )

        y1 = max(
            0,
            min(
                y,
                image_height - 1
            )
        )

        x2 = max(
            x1 + 1,
            min(
                x + w,
                image_width
            )
        )

        y2 = max(
            y1 + 1,
            min(
                y + h,
                image_height
            )
        )

        crop = image[
            y1:y2,
            x1:x2
        ]

        crops[field_name] = crop

        rectangles[field_name] = (
            x1,
            y1,
            x2 - x1,
            y2 - y1
        )

    return crops, rectangles



# ============================================================
# EASY OCR
# ============================================================

def ocr_field(reader, crop, field_name):
    """OCR one field using a field-specific EasyOCR configuration."""
    allowlist = get_allowlist(field_name)

    kwargs = {
        "detail": 1,
        "paragraph": False,
    }

    if allowlist:
        kwargs["allowlist"] = allowlist

    results = reader.readtext(crop, **kwargs)

    if not results:
        return "", 0.0

    results = sorted(
        results,
        key=lambda item: min(point[0] for point in item[0])
    )

    texts = []
    confidences = []

    for _, text, confidence in results:
        text = str(text).strip()
        if text:
            texts.append(text)
            confidences.append(float(confidence))

    if not texts:
        return "", 0.0

    combined_text = " ".join(texts)
    combined_text = normalize_ocr_value(field_name, combined_text)

    average_confidence = sum(confidences) / len(confidences)
    return combined_text, average_confidence


def crop_fields(image, field_definitions, template_width, template_height):
    image_height, image_width = image.shape[:2]

    scale_x = image_width / template_width
    scale_y = image_height / template_height

    crops = {}
    rectangles = {}

    for field_name, field in field_definitions.items():
        x = int(field["x"] * scale_x)
        y = int(field["y"] * scale_y)
        w = int(field["width"] * scale_x)
        h = int(field["height"] * scale_y)

        x1 = max(0, min(x, image_width - 1))
        y1 = max(0, min(y, image_height - 1))
        x2 = max(x1 + 1, min(x + w, image_width))
        y2 = max(y1 + 1, min(y + h, image_height))

        crop = image[y1:y2, x1:x2]

        crops[field_name] = crop
        rectangles[field_name] = (x1, y1, x2 - x1, y2 - y1)

    return crops, rectangles


def scan_image(reader, image, field_definitions, template_width, template_height):
    crops, rectangles = crop_fields(
        image,
        field_definitions,
        template_width,
        template_height
    )

    values = {}
    confidences = {}

    for field_name, crop in crops.items():
        text, confidence = ocr_field(reader, crop, field_name)
        values[field_name] = text
        confidences[field_name] = confidence

    return values, confidences, rectangles


# ============================================================
# DRAW OCR BOXES
# ============================================================

def make_preview(
    image,
    rectangles,
    confidences
):

    preview = image.copy()

    for field_name, (
        x,
        y,
        w,
        h
    ) in rectangles.items():

        confidence = confidences.get(
            field_name,
            0.0
        )

        # --------------------------------------------
        # Confidence colors
        # --------------------------------------------

        if confidence < LOW_CONFIDENCE:

            # RED
            color = (
                0,
                0,
                255
            )

        elif confidence < GOOD_CONFIDENCE:

            # ORANGE
            color = (
                0,
                165,
                255
            )

        else:

            # GREEN
            color = (
                0,
                180,
                0
            )

        # --------------------------------------------
        # Field rectangle
        # --------------------------------------------

        cv2.rectangle(
            preview,
            (x, y),
            (x + w, y + h),
            color,
            3
        )

        # --------------------------------------------
        # Field label
        # --------------------------------------------

        label = (
            f"{field_name}: "
            f"{confidence:.2f}"
        )

        text_y = max(
            25,
            y - 8
        )

        cv2.putText(
            preview,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA
        )

    return preview


def cv_to_pixmap(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    height, width, channels = rgb.shape

    bytes_per_line = (
        channels * width
    )

    qimage = QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888
    )

    return QPixmap.fromImage(
        qimage.copy()
    )


# ============================================================
# SQLITE
# ============================================================

def quote_identifier(name):

    return (
        '"'
        + str(name).replace(
            '"',
            '""'
        )
        + '"'
    )


def create_database(
    json_path,
    database_path,
    field_names
):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    connection = sqlite3.connect(
        database_path
    )

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Create table
    # --------------------------------------------------------

    columns = [

        '"id" INTEGER PRIMARY KEY AUTOINCREMENT',

        '"source_file" TEXT',

        '"page_number" INTEGER',

        '"verified" INTEGER'
    ]

    # Student fields

    for field_name in field_names:

        columns.append(
            f'{quote_identifier(field_name)} TEXT'
        )

    # Confidence columns

    for field_name in field_names:

        columns.append(
            f'{quote_identifier(field_name + "_confidence")} REAL'
        )

    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS students (
            {', '.join(columns)}
        )
        """
    )

    # --------------------------------------------------------
    # Clear previous scan
    # --------------------------------------------------------

    cursor.execute(
        "DELETE FROM students"
    )

    # --------------------------------------------------------
    # Insert data
    # --------------------------------------------------------

    column_names = [

        "source_file",

        "page_number",

        "verified",

        *field_names,

        *(
            field_name + "_confidence"
            for field_name in field_names
        )
    ]

    quoted_columns = ", ".join(
        quote_identifier(column)
        for column in column_names
    )

    placeholders = ", ".join(
        "?"
        for _ in column_names
    )

    for record in data["records"]:

        values = record["values"]

        confidences = record[
            "confidence"
        ]

        row = [

            record["source_file"],

            record["page_number"],

            1
            if record.get(
                "verified",
                False
            )
            else 0
        ]

        # Values

        row.extend(
            values.get(
                field_name,
                ""
            )
            for field_name in field_names
        )

        # Confidence

        row.extend(
            confidences.get(
                field_name,
                0.0
            )
            for field_name in field_names
        )

        cursor.execute(
            f"""
            INSERT INTO students
            ({quoted_columns})
            VALUES
            ({placeholders})
            """,
            row
        )

    connection.commit()

    connection.close()


# ============================================================
# DATABASE EDITOR
# ============================================================

class DatabaseEditor(QWidget):

    def __init__(
        self,
        database_path
    ):

        super().__init__()

        self.database_path = (
            database_path
        )

        layout = QVBoxLayout(
            self
        )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title = QLabel(
            "SQL Database Editor"
        )

        title.setStyleSheet(
            """
            font-size: 20px;
            font-weight: bold;
            """
        )

        layout.addWidget(
            title
        )

        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        self.table = QTableWidget()

        self.table.setAlternatingRowColors(
            True
        )

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        layout.addWidget(
            self.table,
            stretch=3
        )

        # ----------------------------------------------------
        # SQL editor
        # ----------------------------------------------------

        query_label = QLabel(
            "SQL Query"
        )

        layout.addWidget(
            query_label
        )

        self.query = QPlainTextEdit()

        self.query.setPlaceholderText(
            "Example:\n"
            "SELECT * FROM students;"
        )

        self.query.setMaximumHeight(
            120
        )

        layout.addWidget(
            self.query
        )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        buttons = QHBoxLayout()

        execute_button = QPushButton(
            "Execute SQL"
        )

        execute_button.clicked.connect(
            self.execute_query
        )

        buttons.addWidget(
            execute_button
        )

        refresh_button = QPushButton(
            "Refresh Table"
        )

        refresh_button.clicked.connect(
            self.load_table
        )

        buttons.addWidget(
            refresh_button
        )

        layout.addLayout(
            buttons
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.status = QLabel()

        layout.addWidget(
            self.status
        )

        self.load_table()

    def connect(self):

        return sqlite3.connect(
            self.database_path
        )

    # --------------------------------------------------------
    # Load table
    # --------------------------------------------------------

    def load_table(self):

        try:

            connection = self.connect()

            cursor = connection.cursor()

            cursor.execute(
                "SELECT * FROM students"
            )

            rows = cursor.fetchall()

            headers = [
                description[0]
                for description
                in cursor.description
            ]

            connection.close()

            self.table.clear()

            self.table.setColumnCount(
                len(headers)
            )

            self.table.setHorizontalHeaderLabels(
                headers
            )

            self.table.setRowCount(
                len(rows)
            )

            for row_index, row in enumerate(rows):

                for column_index, value in enumerate(row):

                    self.table.setItem(
                        row_index,
                        column_index,
                        QTableWidgetItem(
                            str(value)
                        )
                    )

            self.status.setText(
                f"{len(rows)} student record(s)"
            )

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Database Error",
                str(exc)
            )

    # --------------------------------------------------------
    # Execute SQL
    # --------------------------------------------------------

    def execute_query(self):

        query = (
            self.query
            .toPlainText()
            .strip()
        )

        if not query:
            return

        try:

            connection = self.connect()

            cursor = connection.cursor()

            cursor.execute(
                query
            )

            # ------------------------------------------------
            # SELECT
            # ------------------------------------------------

            if query.lstrip().lower().startswith(
                "select"
            ):

                rows = cursor.fetchall()

                headers = [
                    description[0]
                    for description
                    in cursor.description
                ]

                self.table.clear()

                self.table.setColumnCount(
                    len(headers)
                )

                self.table.setHorizontalHeaderLabels(
                    headers
                )

                self.table.setRowCount(
                    len(rows)
                )

                for row_index, row in enumerate(rows):

                    for column_index, value in enumerate(row):

                        self.table.setItem(
                            row_index,
                            column_index,
                            QTableWidgetItem(
                                str(value)
                            )
                        )

                self.status.setText(
                    f"Query returned "
                    f"{len(rows)} row(s)"
                )

            # ------------------------------------------------
            # INSERT / UPDATE / DELETE etc.
            # ------------------------------------------------

            else:

                connection.commit()

                self.status.setText(
                    "Query executed. "
                    f"{cursor.rowcount} row(s) affected."
                )

                self.load_table()

            connection.close()

        except Exception as exc:

            QMessageBox.critical(
                self,
                "SQL Error",
                str(exc)
            )


# ============================================================
# VERIFICATION WINDOW
# ============================================================

class VerificationWidget(QWidget):

    def __init__(
        self,
        field_names
    ):

        super().__init__()

        self.field_names = (
            field_names
        )

        self.inputs = {}
        self.confidence_labels = {}

        root = QHBoxLayout(
            self
        )

        # ====================================================
        # LEFT SIDE
        # ====================================================

        self.image_label = QLabel()

        self.image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.image_label.setMinimumSize(
            500,
            500
        )

        self.image_label.setStyleSheet(
            "background: #202020;"
        )

        root.addWidget(
            self.image_label,
            stretch=3
        )

        # ====================================================
        # RIGHT SIDE
        # ====================================================

        right = QVBoxLayout()

        self.page_label = QLabel()

        self.page_label.setStyleSheet(
            """
            font-size: 16px;
            font-weight: bold;
            """
        )

        right.addWidget(
            self.page_label
        )

        # ----------------------------------------------------
        # Fields
        # ----------------------------------------------------

        form = QFormLayout()

        for field_name in field_names:

            row = QHBoxLayout()

            line_edit = QLineEdit()

            line_edit.setMinimumWidth(
                240
            )

            confidence = QLabel(
                "0.00"
            )

            confidence.setMinimumWidth(
                55
            )

            confidence.setAlignment(
                Qt.AlignmentFlag.AlignCenter
            )

            row.addWidget(
                line_edit
            )

            row.addWidget(
                confidence
            )

            form.addRow(
                field_name
                .replace(
                    "_",
                    " "
                )
                .title()
                + ":",
                row
            )

            self.inputs[
                field_name
            ] = line_edit

            self.confidence_labels[
                field_name
            ] = confidence

        right.addLayout(
            form
        )

        # ----------------------------------------------------
        # Warning
        # ----------------------------------------------------

        self.warning = QLabel()

        self.warning.setWordWrap(
            True
        )

        right.addWidget(
            self.warning
        )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        button_row = QHBoxLayout()

        self.confirm_button = QPushButton(
            "Confirm & Next"
        )

        self.confirm_button.setMinimumHeight(
            45
        )

        button_row.addWidget(
            self.confirm_button
        )

        self.skip_button = QPushButton(
            "Skip Verification"
        )

        self.skip_button.setMinimumHeight(
            45
        )

        button_row.addWidget(
            self.skip_button
        )

        right.addLayout(
            button_row
        )

        root.addLayout(
            right,
            stretch=2
        )

    # --------------------------------------------------------
    # Display current form
    # --------------------------------------------------------


    def display_record( self,
        image,
        source_file,
        page_number,
        values,
        confidences,
        rectangles
    ):
        self.page_label.setText(
            f"{source_file} | Page {page_number}"
        )

        preview = make_preview(image, rectangles, confidences)
        pixmap = cv_to_pixmap(preview)

        available = self.image_label.size()
        scaled = pixmap.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

        low_confidence_fields = []

        for field_name in self.field_names:
            value = values.get(field_name, "")
            confidence = confidences.get(field_name, 0.0)

            self.inputs[field_name].setText(value)
            self.confidence_labels[field_name].setText(
                f"{confidence:.2f}"
            )

            # Dark grey fill + white text. Confidence is communicated
            # by the border, so text remains readable at every confidence.
            if confidence < LOW_CONFIDENCE:
                border = "#e53935"       # red
                low_confidence_fields.append(field_name)
            elif confidence < GOOD_CONFIDENCE:
                border = "#f9a825"       # yellow
            else:
                border = "#43a047"       # green

            self.inputs[field_name].setStyleSheet(
                f"""
                QLineEdit {{
                    background-color: #303030;
                    color: white;
                    border: 3px solid {border};
                    border-radius: 5px;
                    padding: 5px;
                }}
                """
            )

            self.confidence_labels[field_name].setStyleSheet(
                f"""
                QLabel {{
                    background-color: #303030;
                    color: white;
                    border: 2px solid {border};
                    border-radius: 4px;
                    font-weight: bold;
                    padding: 3px;
                }}
                """
            )

        if low_confidence_fields:
            self.warning.setText(
                "LOW CONFIDENCE: "
                + ", ".join(low_confidence_fields)
                + "\nPlease check these fields carefully."
            )
            self.warning.setStyleSheet(
                "color: #e53935; font-weight: bold;"
            )
        else:
            self.warning.setText(
                "All fields are above the low-confidence threshold."
            )
            self.warning.setStyleSheet("")

    # --------------------------------------------------------
    # Get edited values
    # --------------------------------------------------------

    def get_values(self):

        return {
            field_name:
                self.inputs[
                    field_name
                ].text().strip()

            for field_name
            in self.field_names
        }




# ============================================================
# TEMPLATE FIELD MARKER
# ============================================================

class ImageSelector(QLabel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.press_callback = None
        self.move_callback = None
        self.release_callback = None
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if self.press_callback:
            self.press_callback(event)

    def mouseMoveEvent(self, event):
        if self.move_callback:
            self.move_callback(event)

    def mouseReleaseEvent(self, event):
        if self.release_callback:
            self.release_callback(event)


class FieldMarkerWidget(QWidget):
    """
    Integrated replacement for mark_fields.py.

    The first page of the selected PDF becomes the template. The user
    selects each field by dragging a rectangle and pressing Enter.
    """

    def __init__(self, image, field_names, on_finished, on_cancel):
        super().__init__()
        self.image = image
        self.field_names = field_names
        self.on_finished = on_finished
        self.on_cancel = on_cancel

        self.current_field = 0
        self.fields = {}
        self.start_point = None
        self.end_point = None
        self.display_rect = None
        self.image_label = ImageSelector()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.press_callback = self.mouse_press
        self.image_label.move_callback = self.mouse_move
        self.image_label.release_callback = self.mouse_release

        title = QLabel("Mark Form Fields")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")

        self.instruction = QLabel()
        self.instruction.setWordWrap(True)

        self.accept_button = QPushButton("Accept Field")
        self.accept_button.clicked.connect(self.accept_field)

        redo_button = QPushButton("Redo Current")
        redo_button.clicked.connect(self.redo_field)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.on_cancel)

        buttons = QHBoxLayout()
        buttons.addWidget(self.accept_button)
        buttons.addWidget(redo_button)
        buttons.addStretch()
        buttons.addWidget(cancel_button)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.instruction)
        layout.addWidget(self.image_label, 1)
        layout.addLayout(buttons)

        self.refresh()

    def refresh(self):
        if self.current_field >= len(self.field_names):
            self.instruction.setText(
                "All fields have been marked. Click Finish to save the coordinates."
            )
            self.accept_button.setText("Finish")
        else:
            field = self.field_names[self.current_field]
            self.instruction.setText(
                f"Select: {field}\n"
                "Drag around the complete field area, then click Accept Field. "
                "Coordinates are saved relative to this template."
            )
            self.accept_button.setText("Accept Field")

        self.redraw()

    def redraw(self):
        canvas = self.image.copy()

        for name, data in self.fields.items():
            x, y = data["x"], data["y"]
            w, h = data["width"], data["height"]
            cv2.rectangle(canvas, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(
                canvas, name, (x, max(20, y-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )

        if self.start_point and self.end_point:
            cv2.rectangle(
                canvas, self.start_point, self.end_point, (0, 255, 0), 2
            )

        pixmap = cv_to_pixmap(canvas)
        self.display_rect = self.image_label.contentsRect()
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def widget_to_image(self, pos):
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return None

        label_w = self.image_label.width()
        label_h = self.image_label.height()
        pix_w = pixmap.width()
        pix_h = pixmap.height()

        offset_x = (label_w - pix_w) // 2
        offset_y = (label_h - pix_h) // 2

        px = pos.x() - offset_x
        py = pos.y() - offset_y

        if px < 0 or py < 0 or px >= pix_w or py >= pix_h:
            return None

        image_h, image_w = self.image.shape[:2]
        x = int(px * image_w / pix_w)
        y = int(py * image_h / pix_h)

        return x, y

    def mouse_press(self, event):
        point = self.widget_to_image(event.position().toPoint())
        if point:
            self.start_point = point
            self.end_point = point
            self.redraw()

    def mouse_move(self, event):
        if self.start_point:
            point = self.widget_to_image(event.position().toPoint())
            if point:
                self.end_point = point
                self.redraw()

    def mouse_release(self, event):
        point = self.widget_to_image(event.position().toPoint())
        if point:
            self.end_point = point
            self.redraw()

    def accept_field(self):
        if self.current_field >= len(self.field_names):
            self.on_finished(self.fields)
            return

        if self.start_point is None or self.end_point is None:
            QMessageBox.warning(self, "No selection", "Drag a rectangle first.")
            return

        x1, y1 = self.start_point
        x2, y2 = self.end_point
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2-x1), abs(y2-y1)

        if w < 5 or h < 5:
            QMessageBox.warning(self, "Selection too small", "Select a larger field area.")
            return

        self.fields[self.field_names[self.current_field]] = {
            "x": x, "y": y, "width": w, "height": h
        }

        self.current_field += 1
        self.start_point = None
        self.end_point = None
        self.refresh()

    def redo_field(self):
        if self.current_field == 0:
            return
        self.current_field -= 1
        self.fields.pop(self.field_names[self.current_field], None)
        self.start_point = None
        self.end_point = None
        self.refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.redraw()


# ============================================================
# OCR PROGRESS SCREEN
# ============================================================

class OCRProgressWidget(QWidget):

    def __init__(self, total):
        super().__init__()
        self.total = max(1, total)

        title = QLabel("Scanning Forms")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        self.status = QLabel()
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress = QProgressBar()
        self.progress.setRange(0, self.total)
        self.progress.setValue(0)
        self.progress.setMinimumHeight(28)

        self.detail = QLabel("Preparing OCR...")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.detail)
        layout.addStretch()

    def update_progress(self, current, total, filename, page):
        self.progress.setMaximum(total)
        self.progress.setValue(current)
        self.status.setText(f"Scanning document {current} of {total}")
        self.detail.setText(f"{filename} — page {page}")


# ============================================================
# MAIN WINDOW
# ============================================================



def load_template_image(path):
    path = Path(path)

    if path.suffix.lower() == ".pdf":
        document = pymupdf.open(str(path))
        try:
            if document.page_count == 0:
                raise RuntimeError("The PDF contains no pages.")
            return image_from_pdf_page(document[0])
        finally:
            document.close()

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Form Scanner OCR")
        self.resize(1400, 850)

        self.field_definitions = {}
        self.field_names = list(FIELD_TYPES.keys())
        self.template_width = 1
        self.template_height = 1

        self.reader = None
        self.records = []
        self.scanned_results = []
        self.verification_index = 0
        self.input_path = None
        self.images = []

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.setup_start_page()

    def setup_start_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Form Scanner OCR")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold;")
        layout.addWidget(title)

        description = QLabel(
            "Select an image or PDF containing filled forms.\n\n"
            "For a new form layout, the first PDF page is used as the template "
            "and you will mark each field once."
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)

        layout.addSpacing(30)

        open_button = QPushButton("Open Image / PDF")
        open_button.setMinimumHeight(60)
        open_button.clicked.connect(self.open_input)
        layout.addWidget(open_button)

        self.start_status = QLabel()
        self.start_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.start_status.setWordWrap(True)
        layout.addWidget(self.start_status)

        layout.addStretch()
        self.stack.addWidget(page)

    def open_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Form Image or PDF",
            "",
            "Images / PDF (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pdf)"
        )

        if not path:
            return

        try:
            self.prepare_scan(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not start scan", str(exc))



    def prepare_scan(self, path):
        self.input_path = path
        self.images = []

        self.start_status.setText("Loading first page as template...")
        QApplication.processEvents()

        template_image = load_template_image(path)
        save_template_image(template_image)

        existing = Path(FIELDS_JSON).exists()

        if existing:
            reply = QMessageBox.question(
                self,
                "Existing field coordinates",
                "Existing field coordinates were found.\n\n"
                "Use the existing coordinates, or mark the fields again?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.start_ocr()
                return

        self.start_field_marker(template_image)

    def start_field_marker(self, template_image):
        self.marker = FieldMarkerWidget(
            template_image,
            self.field_names,
            self.finish_field_marker,
            self.cancel_to_start
        )
        self.stack.addWidget(self.marker)
        self.stack.setCurrentWidget(self.marker)

    def finish_field_marker(self, fields):
        self.field_definitions = fields
        save_field_definitions(FIELDS_JSON, fields)
        self.start_ocr()

    def cancel_to_start(self):
        self.stack.setCurrentIndex(0)

    def start_ocr(self):
        self.field_definitions = load_field_definitions(FIELDS_JSON)
        self.field_names = list(self.field_definitions.keys())

        self.start_status.setText("Loading document pages...")
        QApplication.processEvents()
        self.images = load_input(self.input_path)

        if not self.images:
            raise RuntimeError("No pages/images found.")

        self.template_width, self.template_height = template_image_size(
            self.images[0][2]
        )

        self.start_status.setText("Loading EasyOCR model...")
        QApplication.processEvents()

        if self.reader is None:
            self.reader = easyocr.Reader(
                OCR_LANGUAGES,
                gpu=OCR_GPU,
                verbose=False
            )

        self.records = []
        self.scanned_results = []
        self.verification_index = 0

        self.progress = OCRProgressWidget(len(self.images))
        self.stack.addWidget(self.progress)
        self.stack.setCurrentWidget(self.progress)

        self.run_full_ocr()

    def run_full_ocr(self):
        total = len(self.images)

        for index, (source_file, page_number, image) in enumerate(
            self.images, start=1
        ):
            self.progress.update_progress(
                index, total, source_file, page_number
            )
            self.statusBar().showMessage(
                f"OCR: {source_file}, page {page_number} ({index}/{total})"
            )
            QApplication.processEvents()

            try:
                values, confidences, rectangles = scan_image(
                    self.reader,
                    image,
                    self.field_definitions,
                    self.template_width,
                    self.template_height
                )
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "OCR Error",
                    f"Error on {source_file}, page {page_number}:\n{exc}"
                )
                return

            self.scanned_results.append({
                "source_file": source_file,
                "page_number": page_number,
                "image": image,
                "values": values,
                "confidence": confidences,
                "rectangles": rectangles,
            })

            self.progress.update_progress(
                index, total, source_file, page_number
            )
            QApplication.processEvents()

        self.progress.status.setText("OCR complete")
        self.progress.detail.setText(
            f"{total} document page(s) scanned. Opening verification..."
        )
        self.progress.progress.setValue(total)
        QApplication.processEvents()

        self.start_verification()

    def start_verification(self):
        if not self.scanned_results:
            QMessageBox.warning(
                self, "No data", "No OCR results were created."
            )
            self.stack.setCurrentIndex(0)
            return

        self.verification = VerificationWidget(self.field_names)
        self.verification.confirm_button.clicked.connect(self.confirm_current)
        self.verification.skip_button.clicked.connect(self.skip_current)

        self.stack.addWidget(self.verification)
        self.verification_index = 0
        self.show_verification_record()

    def show_verification_record(self):
        result = self.scanned_results[self.verification_index]

        self.verification.display_record(
            result["image"],
            result["source_file"],
            result["page_number"],
            result["values"],
            result["confidence"],
            result["rectangles"]
        )

        self.statusBar().showMessage(
            f"Verification: {self.verification_index + 1}/"
            f"{len(self.scanned_results)}"
        )
        self.stack.setCurrentWidget(self.verification)

    def confirm_current(self):
        result = self.scanned_results[self.verification_index]

        self.records.append({
            "source_file": result["source_file"],
            "page_number": result["page_number"],
            "values": self.verification.get_values(),
            "confidence": result["confidence"],
            "verified": True
        })

        self.next_verification()

    def skip_current(self):
        result = self.scanned_results[self.verification_index]

        self.records.append({
            "source_file": result["source_file"],
            "page_number": result["page_number"],
            "values": result["values"],
            "confidence": result["confidence"],
            "verified": False
        })

        self.next_verification()

    def next_verification(self):
        self.verification_index += 1

        if self.verification_index >= len(self.scanned_results):
            self.finish_scan()
        else:
            self.show_verification_record()

    def finish_scan(self):
        if not self.records:
            QMessageBox.warning(self, "No data", "No records were created.")
            return

        data = {
            "created_at": datetime.now().isoformat(),
            "source": str(self.input_path),
            "fields": self.field_names,
            "records": self.records
        }

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        create_database(
            OUTPUT_JSON,
            OUTPUT_DB,
            self.field_names
        )

        editor = DatabaseEditor(OUTPUT_DB)
        self.stack.addWidget(editor)
        self.stack.setCurrentWidget(editor)

        self.statusBar().showMessage(
            f"Finished. JSON: {OUTPUT_JSON} | DB: {OUTPUT_DB}"
        )

        QMessageBox.information(
            self,
            "Scan Complete",
            f"Processed {len(self.records)} record(s).\n\n"
            f"JSON:\n{OUTPUT_JSON}\n\n"
            f"SQLite:\n{OUTPUT_DB}"
        )


def template_image_size(image):
    height, width = image.shape[:2]
    return width, height


# ============================================================
# RUN
# ============================================================

def main():

    app = QApplication(
        sys.argv
    )

    window = MainWindow()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()
