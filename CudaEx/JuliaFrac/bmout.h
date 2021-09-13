#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <windows.h>



bool SaveImage(const std::string& szPathName, const char *lpBits, int w, int h) {
    // Create a new file for writing
    std::ofstream pFile(szPathName, std::ios_base::binary);
    if (!pFile.is_open()) {
        return false;
    }

    BITMAPINFOHEADER bmih;
    bmih.biSize = sizeof(BITMAPINFOHEADER);
    bmih.biWidth = w;
    bmih.biHeight = h;
    bmih.biPlanes = 1;
    bmih.biBitCount = 24;
    bmih.biCompression = BI_RGB;
    bmih.biSizeImage = w * h * 3;

    BITMAPFILEHEADER bmfh;  
    int nBitsOffset = sizeof(BITMAPFILEHEADER) + bmih.biSize;
    LONG lImageSize = bmih.biSizeImage;
    LONG lFileSize = nBitsOffset + lImageSize;
    bmfh.bfType = 'B' + ('M' << 8);
    bmfh.bfOffBits = nBitsOffset;
    bmfh.bfSize = lFileSize;
    bmfh.bfReserved1 = bmfh.bfReserved2 = 0;
    struct RGBQUAD {
        uint8_t rgbtBlue;
        uint8_t rgbtGreen;
        uint8_t rgbtRed;
        uint8_t rgbtDark;
    }RGBQUAD;

    // Write the bitmap file header
    pFile.write((const char*)&bmfh, sizeof(BITMAPFILEHEADER));
    UINT nWrittenFileHeaderSize = pFile.tellp();

    // And then the bitmap info header
    pFile.write((const char*)&bmih, sizeof(BITMAPINFOHEADER));
    UINT nWrittenInfoHeaderSize = pFile.tellp();

    pFile.write((const char*)&RGBQUAD, sizeof(RGBQUAD));
    // Finally, write the image data itself
    //-- the data represents our drawing
    pFile.write(lpBits, w * h  * 4);
    UINT nWrittenDIBDataSize = pFile.tellp();
    pFile.close();

    return true;
}