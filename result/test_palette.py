from __future__ import division
import Image

palette = []
levels = 8
stepsize = 256 // levels 
for i in range(256):
    v = i // stepsize * stepsize 
    palette.extend((v,v,v))

assert len(palette) == 768

original_path = 'street_urb19.jpg'
original = Image.open(original_path)
converted = Image.new('P',original.size)
converted.putpalette(palette)
converted.paste(original,(0,0))
print converted.palette
converted.convert('RGB').save('converted.jpg')
#converted.show()


