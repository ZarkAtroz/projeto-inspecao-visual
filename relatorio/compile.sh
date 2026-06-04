#!/bin/bash
# Compilar o relatório LaTeX (duas passadas para referências cruzadas e sumário)
cd "$(dirname "$0")"
pdflatex relatorio.tex
pdflatex relatorio.tex
echo ""
echo "Compilacao concluida: relatorio.pdf"
