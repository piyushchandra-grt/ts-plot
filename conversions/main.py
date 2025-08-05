import aspose.pdf as ap

input_pdf = "conversions\one.pdf"
output_tex = "pdf_to_tex.tex"

# Open PDF file
document = ap.Document(input_pdf)

# Create an object of LaTeXSaveOptions class
saveOptions = ap.LaTeXSaveOptions()

# Save PDF as TEX
document.save(output_tex, saveOptions)