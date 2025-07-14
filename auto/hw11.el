;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "hw11"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("report" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fullpage" "") ("amsmath" "") ("amssymb" "") ("bm" "") ("upgreek" "") ("mathrsfs" "") ("algorithmic" "") ("algorithm" "") ("graphicx" "") ("subcaption" "") ("setspace" "") ("color" "") ("multirow" "") ("alltt" "") ("cancel" "") ("listings" "")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-environments-local "alltt")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "report"
    "rep12"
    "fullpage"
    "amsmath"
    "amssymb"
    "bm"
    "upgreek"
    "mathrsfs"
    "algorithmic"
    "algorithm"
    "graphicx"
    "subcaption"
    "setspace"
    "color"
    "multirow"
    "alltt"
    "cancel"
    "listings")
   (TeX-add-symbols
    '("V" 1)
    '("Cov" 2)
    '("E" 1)
    "N"
    "U"
    "Poi"
    "Exp"
    "G"
    "Ber"
    "Lap"
    "btheta"
    "bSigma"))
 :latex)

