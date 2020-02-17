# pip install PyMuPDF
import fitz
    
def pdf_to_img(pdffile, outpath, zoom = 1):
  
    doc = fitz.open(pdffile) 

    for page_n in range(len(doc)):
        page = doc.loadPage(page_n) #number of page
        mat = fitz.Matrix(zoom, zoom)
        pix = page.getPixmap(matrix = mat, alpha = False)
        imgname = 'page_' + str(page_n) + '.jpg'
        pix.writePNG(outpath + imgname)
        #pix.writeImage(outpath + imgname)
    doc.close()
