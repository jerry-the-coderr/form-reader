# Form Scanner OCR

A basic desktop application for scanning filled forms using OCR, verifying the extracted information, and saving the results to JSON and SQLite.

> **Project status:** Experimental / personal project
> **Limitation:** This is designed for basic, structured forms and is not intended to be a general-purpose document OCR system.

## What It Does

The program follows this basic workflow:

1. Select an image or PDF containing a filled form.
2. If necessary, select the locations of individual fields on the form.
3. Convert PDF pages into images.
4. Crop the defined fields.
5. Run OCR on each field using EasyOCR.
6. Apply field-specific OCR restrictions and basic text normalization.
7. Display the OCR results with confidence indicators.
8. Allow the user to verify or edit the extracted values.
9. Save the verified results to:

   * `scan_results.json`
   * `scan_results.db`
10. Provide a simple SQLite database editor.

The application uses a PySide6 graphical interface rather than a command-line interface.

---

## Technologies Used

| Technology         | Purpose                                                   |
| ------------------ | --------------------------------------------------------- |
| **Python**         | Main programming language                                 |
| **PySide6**        | Desktop graphical user interface                          |
| **OpenCV (`cv2`)** | Image processing, cropping, drawing, and image conversion |
| **NumPy**          | Image array manipulation                                  |
| **EasyOCR**        | Optical Character Recognition                             |
| **PyMuPDF**        | Converting PDF pages into images                          |
| **SQLite3**        | Storing scanned records in a database                     |
| **JSON**           | Storing field definitions and scan results                |
| **pathlib**        | File and path handling                                    |

### What each major component is used for

**PySide6** is used to create the application windows, buttons, text fields, progress screen, verification interface, and database editor.

**OpenCV** is used for reading images, cropping form fields, drawing field rectangles, and preparing images for display.

**EasyOCR** is used to read text from the cropped form fields and provide OCR confidence scores.

**PyMuPDF** is used to render PDF pages into images before they are processed by OpenCV and EasyOCR.

**SQLite** is used to store the final extracted records in a local database.

**JSON** is used for storing field coordinates and the final OCR results.

---

# DevNote:

```
Idk why this is but while installing [easyocr], the first attempt always gives an error which solves after manually installing [numpy] and [triton] first followed by reinstalling [easyocr]
```

# Installation

## 1. Clone or download the project

Download the project and open a terminal in the project directory.

Example:

```bash
mkdir form-scanner-ocr
cd form-scanner-ocr
```

---

## 2. Create a virtual environment

This is recommended so the project's dependencies do not interfere with other Python projects.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install dependencies using `requirements.txt`

Make sure the project contains a `requirements.txt` file.

Then run:

```bash
pip install -r requirements.txt
```

This installs the Python packages required by the program.``````````````

The standard-library modules used by the project, such as `json`, `sqlite3`, `pathlib`, `datetime`, and `sys`, do **not** need to be installed separately.

---

## 4. Run the program

After installing the dependencies:

```bash
python main.py
```

If your file has a different name, replace `main.py` with the actual filename.

---

# Basic Usage

## Step 1 — Select a form

Click:

**Open Image / PDF**

The program accepts common image formats as well as PDF files.

---

## Step 2 — Define the fields

For a new form layout, the program uses the first page as a template.

You can drag a rectangle around each field, such as:

* Name
* Class
* Section
* Roll Number
* Phone Number
* Blood Group

The coordinates are saved so they can be reused for subsequent scans.

---

## Step 3 — OCR

The program processes each page and:

```text
PDF/Image
    ↓
OpenCV Image
    ↓
Crop individual fields
    ↓
EasyOCR
    ↓
Extracted text + confidence
```

Different fields can have different OCR restrictions. For example, numeric fields are restricted to digits, while blood-group fields use an appropriate character allowlist.

---

## Step 4 — Verify the results

The verification screen displays:

* The scanned form
* The detected field values
* OCR confidence
* Confidence-based borders

The field borders use:

* 🔴 **Red** — low confidence
* 🟡 **Yellow** — medium confidence
* 🟢 **Green** — good confidence

The extracted values can be edited before confirming them.

---

## Step 5 — Results

After verification, the program saves the results as:

```text
scan_results.json
scan_results.db
```

The JSON file contains the scan results and metadata.

The SQLite database contains the extracted student/form information and OCR confidence values.

---

# Important Limitations

This project **only works reliably for basic, structured forms**.

It is not designed to handle arbitrary documents.

For example, it works best when:

* The form has a consistent layout.
* The fields are in predictable locations.
* The writing is reasonably clear.
* The input image is reasonably readable.
* The fields can be identified with rectangular regions.
* The same general form layout is used across scanned pages.

It is **not** a general-purpose OCR system capable of automatically understanding any document.

### Other limitations

* Handwriting recognition can be unreliable.
* Poor-quality scans can produce incorrect OCR results.
* Rotated or heavily distorted forms may cause problems.
* The field coordinates are layout-dependent.
* OCR confidence does not guarantee that the extracted value is correct.
* The application currently relies on manual field selection for a new form layout.
* The OCR process runs on the CPU (`gpu=False`), so processing can be relatively slow on low-end hardware.

---

# Project Structure

A simplified view of the program is:

```text
main.py
│
├── Configuration
│
├── Field definitions
│
├── PDF → Image conversion
│
├── Field cropping
│
├── EasyOCR processing
│
├── OCR preview
│
├── SQLite database creation
│
├── Database editor
│
├── Verification window
│
├── Field selection window
│
├── OCR progress window
│
└── Main application window
```

The main application coordinates these components and controls the overall workflow.

---

# Development Note

This project was **mostly vibe coded**.

Before arriving at this version, I attempted to build the project from scratch **three times**.

Those attempts helped establish what the application needed, but this version was developed heavily through AI-assisted coding, experimentation, modification, and debugging rather than being written completely from scratch by hand.

As a result:

* Some parts of the code are cleaner than others.
* There may be redundant or overly complicated sections.
* The architecture is not necessarily how I would design the project from the beginning today.
* The project should be treated as an experimental learning project rather than production-ready software.

The goal was primarily to **get the concept working**, learn from the implementation, and iterate on it.

---

# Why This Project Exists

The project was created as an experiment in combining:

* Python
* OCR
* Computer vision
* PDF processing
* GUI development
* Databases

into one practical application.

The project is also an attempt to understand how a larger Python application can be structured instead of working only with small standalone scripts.

---
