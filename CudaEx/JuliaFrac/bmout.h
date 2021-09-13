#pragma once
#include<fstream>
#include <iostream>
using namespace std;

//Set structure packing mode to 2 Bytes
//It prevents structure for being extended by additional alignment Bytes
#pragma pack(push,2)
//C++11 fixed width integer types are used
struct BITMAPFILEHEADER {
    uint16_t bfType;
    uint32_t bfSize;
    uint32_t bfReserved1;
    uint32_t bfOffBits;
};

//C++11 fixed width integer types are used
struct BITMAPINFOHEADER {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};

//C++11 fixed width integer types are used
struct RGBTRIPLE {
    uint8_t rgbtBlue;
    uint8_t rgbtGreen;
    uint8_t rgbtRed;
};

int main()
{
    BITMAPFILEHEADER bfh;
    BITMAPINFOHEADER bfi;
    char filename[20];
    cout << "enter filename";
    cin >> filename;

    ifstream fin(filename, ios::binary);
    fin.read((char*)&bfh, 14);
    fin.read((char*)&bfi, 40);

    int width = bfi.biWidth;
    int height = bfi.biHeight;

    //determine line size in bytes
    //RGBTRIPLE bytes (width*3) + additional trash of (width%4) bytes
    int line = width * 3 + width % 4;
    //image size (should be equal to bfi.biSize)
    int size = line * height;

    //Pixels array that stores rgb triples
    //Important fact!!!: image is a TOP-DOWN. You should reverse rows order
    RGBTRIPLE** pixels = new RGBTRIPLE * [height];

    /*
    //Fast greedy variant of pixels storage and processing
    //The whole image is stored in the last row
    //Other rows store pointers to corresponding positions inside that row
    pixels[height-1] = (RGBTRIPLE*)(new char[size]);
    fin.read((char*)pixels[height-1], size);
    //Climb from bottom to the top and assign adresses (shifted pointers)
    for (int i = height - 2; i >= 0; i--)
        //Here we shift each row exactly by 'line' bytes regarding to the previous row address
        pixels[i] = (RGBTRIPLE*)((char*)pixels[i + 1] + line);
    */

    //Usual variant
    //Every row stores it's portion of image data
    for (int i = height - 1; i >= 0; i--)
    {
        pixels[i] = (RGBTRIPLE*)(new char[size]);
        fin.read((char*)pixels[i], line);
    }

    fin.close();

    //!!!
    //Add 10x10 square to the left upper corner ot the image
    //You should implement your image processing instead of this code between //!!!'s
    int hl, hr, wr, k, l, temp_k;
    RGBTRIPLE temp;
    cout << "enter heightleft" << endl;
    cin >> hl;
    cout << "enter heightright" << endl;
    cin >> hr;
    cout << "enter widthright" << endl;
    cin >> wr;
    cin >> k;
    l = k;
    temp_k = k;
    for (int i = hl; i < hr; i++)
        for (int j = l; j < wr; j += l) {
            temp = pixels[i][j];
            while (k >= 0) {
                pixels[i][j - k] = temp;
                k--;
            }
            k = temp_k;
        }

    //!!!


    ofstream fout("output.bmp", ios::binary);
    fout.write((char*)&bfh, sizeof(bfh));
    fout.write((char*)&bfi, sizeof(bfi));

    /*
    //Writing pixels to file and memory deallocation for fast&greedy variant
    //In this case image rows order does not matter. Physically the whole image is stored in one row.
    fout.write((char*)pixels[height - 1], size);

    delete[]pixels[height - 1];
    delete[]pixels;
    */

    //Writing pixels to file and deallocating memory
    for (int i = height - 1; i >= 0; i--)
    {
        fout.write((char*)pixels[i], line);
        delete[]pixels[i];
    }
    delete[]pixels;

    fout.close();
    return 0;
}

