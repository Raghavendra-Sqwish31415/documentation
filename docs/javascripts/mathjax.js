window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['ams', 'color', 'configmacros']},
    macros: {
      textbf: ["{\\bf #1}", 1],
      mathbb: ["{\\bf #1}", 1],
      arg: "\\mathop{\\rm arg}\\nolimits",
      max: "\\mathop{\\rm max}\\nolimits",
      min: "\\mathop{\\rm min}\\nolimits"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => { 
  MathJax.typesetPromise()
}) 