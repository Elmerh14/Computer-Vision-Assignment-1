from PIL import Image
import numpy as np 

#load the input image
img = Image.open("8-bit-graphics-pixels-scene-with-village.jpg")
#load direct access to the pixels at each container
pixels = img.load()

#get the widht and the height of the image 
width, height = img.size
#create an empty grey scale container each pixel will store a value from 0 -255
greyScaleImage = Image.new("L", (width, height))
#get direct access to each pixel in the empty container
greyPixels = greyScaleImage.load()
#travers throgh every pixel in the image
for x in range(width):
    for y in range(height):
        #unpack the values rgb value at each cordinate in the image
        r, g, b = pixels[x,y]
        #us the luminance formula to calculate the brightness value 
        greyValue = int(0.299 * r + 0.587 * g + 0.114 * b)
        #assign the intensity to every pixel in grey scal image
        greyPixels[x,y] = greyValue

img.show()
greyScaleImage.show()

# part 2

def addNoise(image, sigma):
    imgArray = np.array(image, dtype=np.float32)
    n = np.random.normal(0, sigma, imgArray.shape)

    noisyImage = np.clip(imgArray + n, 0 , 255)

    noisyImage = noisyImage.astype(np.uint8)

    return Image.fromarray(noisyImage, mode="L")

sigmas = [1, 10, 30, 50]
for s in sigmas:
    noisy = addNoise(greyScaleImage, s)
    noisy.show()

#part 3 
def addSaltAndPepper(image, noisePercentage):
    imgArray = np.array(image, dtype=np.uint8)
    output = np.copy(imgArray)

    totalPixels = imgArray.size

    numNosie = int(totalPixels * noisePercentage)

    #generate random cordinates for salt and pepper
    coordinates = [(np.random.randint(0, height -1), np.random.randint(0, width -1)) for _ in range(numNosie)]
    
    #split in half
    half = numNosie // 2
    saltCoordinates = coordinates[:half]
    pepperCoordinates = coordinates[half:]

    #apply
    [output.__setitem__(coord, 255) for coord in saltCoordinates]
    [output.__setitem__(coord, 0) for coord in pepperCoordinates]

    return Image.fromarray(output, mode="L")

noisy10 = addSaltAndPepper(greyScaleImage, 0.10)
noisy30 = addSaltAndPepper(greyScaleImage, 0.30)

noisy10.show()
noisy30.show()

#part 4
def boxFilter(image, kernalSize=3):
    imgArray = np.array(image, dtype=np.float32)
    pad = kernalSize // 2
    padded = np.pad(imgArray, pad, mode='edge')
    output = np.zeros_like(imgArray, dtype=np.float32)

    for i in range(imgArray.shape[0]):
        for j in range(imgArray.shape[1]):
            region = padded[i:i+kernalSize, j:j+kernalSize]
            output[i, j] = np.mean(region)

    return Image.fromarray(output.astype(np.uint8))


def medianFilter(image, kernalSize=3):
    imgArray = np.array(image, dtype=np.float32)
    pad = kernalSize // 2
    padded = np.pad(imgArray, pad, mode='edge')
    output = np.zeros_like(imgArray, dtype=np.float32)

    for i in range(imgArray.shape[0]):
        for j in range(imgArray.shape[1]):
            region = padded[i:i+kernalSize, j:j+kernalSize]
            output[i, j] = np.median(region)

    return Image.fromarray(output.astype(np.uint8))


def gaussianFiliter(image, kernalSize=3, sigma=1):
    imgArray = np.array(image, dtype=np.float32)
    pad = kernalSize // 2
    padded = np.pad(imgArray, pad, mode='edge')
    output = np.zeros_like(imgArray, dtype=np.float32)

    # Build the Gaussian kernel
    k = kernalSize // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    gaussianKernal = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussianKernal /= np.sum(gaussianKernal)

    # Apply kernel
    for i in range(imgArray.shape[0]):
        for j in range(imgArray.shape[1]):
            region = padded[i:i+kernalSize, j:j+kernalSize]
            output[i, j] = np.sum(region * gaussianKernal)

    return Image.fromarray(output.astype(np.uint8))

# ----------------------------
# Apply filters directly to your noisy images
# ----------------------------

noisyGaussian = addNoise(greyScaleImage, 50)
noisySP = addSaltAndPepper(greyScaleImage, 0.3)

# Apply filters
noisBoxFiltered = boxFilter(noisyGaussian)
noiseMedianFiltered = medianFilter(noisySP)
noiseGaussianFiltered = gaussianFiliter(noisyGaussian, sigma=1)

spBoxFiltered = boxFilter(noisySP)
spMedianFiltered = medianFilter(noisySP)
spGaussianFiltered = gaussianFiliter(noisySP, sigma=1)

noisBoxFiltered.show()
noiseMedianFiltered.show()
noiseGaussianFiltered.show()

spBoxFiltered.show()
spMedianFiltered.show()
spGaussianFiltered.show()

