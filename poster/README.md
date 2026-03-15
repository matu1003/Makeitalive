# LaTeX Posters for Makeitalive

This folder contains the LaTeX source files for the Makeitalive project presentation.

---

## 🛠️ How to Modify the Posters

Here is a guide to customizing the content once you have opened the `.tex` file (on Overleaf or locally).

### 1. Structure and Columns
The posters are configured in **Portrait** (A0) with **2 columns**.
- Width is managed by `\column{0.5}`. 
- To adjust the distribution (e.g., a wider column), change it to, for example, `\column{0.4}` and `\column{0.6}`.

### 2. Visual Style
- **Theme**: `\usetheme{Rays}` defines the overall look. Other options: `Wave`, `Simple`, `Default`.
- **Colors**: `\usecolorstyle{Default}`. You can try `BlueGrayOrange`, `GreenMarine`, or `RedWhiteBlue`.

### 3. Adding/Modifying Content
Each section is a block:
```latex
\block{Section Title}{
    Your text here...
}
```

### 4. Diagrams and Drawings (TikZ)
Poster 1 contains a diagram of the U-Net.
- Look for `\begin{tikzpicture}`.
- `node distance=1.5cm` manages the vertical spacing between layers.
- `scale=0.8` allows you to resize the entire diagram without changing the proportions.

### 5. Compilation
Always use **pdflatex**. 
- **Watch out for Underscores**: Characters like `_` must be preceded by a backslash `\_` if they are in normal text, or placed between dollar signs `$x_i$` if they are mathematical.

---

## 🚀 Generate PDFs

```bash
pdflatex poster1_v2.tex
pdflatex poster2_v2.tex
```