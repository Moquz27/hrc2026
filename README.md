# Surface Plotter

## 1. Project Overview

This project is a Python program for visualizing a 3D surface of the form:

```text
z = f(x,y)
```

The surface is drawn above a projection region in the `xOy` plane:

```text
D = {(x,y) | c <= y <= d and g(y) <= x <= h(y)}
```

The user enters the function `f(x,y)`, the boundary curves `g(y)` and `h(y)`,
and the constants `c` and `d`.

## 2. Repository Structure

```text
surface_plot_submission/
├── README.md
├── requirements.txt
├── surface_plot.py
└── outputs/          # created automatically after running successfully
```

- `README.md`: project explanation and usage instructions.
- `requirements.txt`: required Python packages.
- `surface_plot.py`: main terminal program.
- `outputs/`: stores generated plot images and input records.

## 3. Requirements

- Python 3.10 or newer recommended
- numpy
- matplotlib
- sympy

## 4. Installation

First, go to the submission folder:

```bash
cd surface_plot_submission
```

Windows:

```bat
python -m pip install -r requirements.txt
```

macOS/Linux:

Using a virtual environment is recommended on macOS/Linux to avoid system
Python or Homebrew Python permission issues:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If you already have a working Python environment, direct installation is also
possible:

```bash
python3 -m pip install -r requirements.txt
```

## 5. How to Run

Make sure you are inside the `surface_plot_submission/` folder.

### Windows

```bat
python surface_plot.py
```

### macOS/Linux

If you use a virtual environment, activate it first:

```bash
source .venv/bin/activate
```

Then run:

```bash
python3 surface_plot.py
```

## 6. Input Rules

- Use Python math syntax.
- Use `x**2` instead of `x^2`.
- `f(x,y)` may use the variables `x` and `y`.
- `g(y)` and `h(y)` may only use the variable `y`.
- `c` and `d` must be numbers.
- `c < d`.
- `g(y) <= h(y)` on the interval `[c,d]`.

Example input:

```text
f(x,y): x**2 + y**2
g(y): y
h(y): y + 2
c: 0
d: 2
```

Supported common functions and constants include:
`sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `Abs`, `pi`, and `E`.

## 7. Mathematical Method

The projection region is:

```text
D = {(x,y) | c <= y <= d and g(y) <= x <= h(y)}
```

The program samples `y` in `[c,d]`. For each value of `y`, it generates `x`
using:

```text
x = g(y) + t(h(y)-g(y)), 0 <= t <= 1
```

Then it computes:

```text
z = f(x,y)
```

This creates a rectangular numerical grid that represents the curved region
`D`.

## 8. Output

After successful input, the program:

- opens a 3D Matplotlib chart window,
- automatically creates the `outputs/` folder,
- saves the chart as a `.png` file,
- saves the entered input values as a matching `.txt` file.

The `outputs/` folder is created automatically only after the program runs
successfully and saves output files.

Output filenames include a timestamp, for example:

```text
surface_20260511_153020.png
surface_20260511_153020.txt
```

## 9. Notes / Limitations

- The program plots surfaces in the explicit form `z = f(x,y)`.
- It does not plot full implicit surfaces directly, such as
  `x^2 + y^2 + z^2 = 1`.
- A complete sphere cannot be represented as a single explicit function
  `z = f(x,y)`. It must be split into an upper half and a lower half.
- For a sphere, the user can plot the upper half using:
  `f(x,y) = sqrt(1 - x**2 - y**2)`.
- Some functions may be undefined on parts of the region, which can cause an
  invalid input error.
- Use valid Python-style mathematical expressions.
