#pragma once

#include <iostream>
#include <random>
#include <utility>

using namespace std;

namespace mymatrix {
    // шаблонный класс Матрица
    template <typename T>
    class MATRIX
    {
    private:
        T* M; // матрица
        uint32_t rows; // количество строк
        uint32_t columns; // количество столбцов

    public:
        static void initRand(uint32_t seed)
        {
            srand(seed);
        }
        // конструкторы
        MATRIX() :rows(0), columns(0), M(nullptr) {}

        // конструктор с двумя параметрами
        MATRIX(int _rows, int _columns)
        {
            rows = _rows;
            columns = _columns;

            // Выделить память для матрицы
            // Выделить пам'ять для массива указателей
            M = (T*) new T[rows * columns]; // количество строк * количество столбцов
        }

        // Конструктор копирования - обязательный
        MATRIX(const MATRIX& _M)
        {
            // Создается новый объект для которого виделяется память
            // Копирование данных *this <= _M
            rows = _M.rows;
            columns = _M.columns;

            // Выделить память для M
            M = (T*) new T [rows * columns]; // количество строк, количество указателей

            // заполнить значениями
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < columns; j++)
                    M[j + columns * i] = _M.M[j + columns * i];
        }

        MATRIX(MATRIX&& _M)
        {
            // Создается новый объект для которого виделяется память
            // Копирование данных *this <= _M
            rows = _M.rows;
            columns = _M.columns;

            // Выделить память для M
            M = _M.M; 
            _M.M = nullptr;
        }

        T* get_ptr()
        {
            return M;
        }

        
        MATRIX& RandInsert()
        {
            if (!empty())
            {
                
                for (size_t i = 0; i < rows; i++)
                    for (size_t j = 0; j < columns; j++) 
                    {
                        T tmp = (T)(rand() % 256 + 2 * i - j);
                        this->Set(i, j, tmp);
                    }

                return *this;
            }
            else 
            {
                printf("Matrix Empty ");
                return *this;
            }
        }

        void erase()
        {
            if (!empty())
            {
                delete[] M;
            }
            this->rows = 0;
            this->columns = 0;
            this->M = nullptr;
            return;
        }

        //проверка матрицы на пустоту
        bool empty()
        {
            return  !((rows > 0) && (columns > 0) && (M!=nullptr));
        }

        // методы доступа
        T At(int i, int j)
        {
            if (!empty() &&
                i < rows &&
                j < columns)
                return M[j + columns * i];
            else
                return 0;
        }

        void Set(int i, int j, const T& value)
        {
            if ((i < 0) || (i >= rows))
                return;
            if ((j < 0) || (j >= columns))
                return;
            M[j + columns * i] = value;
        }

        // метод, выводящий матрицу
        MATRIX& Print(const char* ObjName = "")
        {
            cout << "Object: " << ObjName << endl;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                    cout << M[j + columns * i] << "  ";
                cout << endl;
            }
            cout << endl << endl;
            return *this;
        }

        const MATRIX& operator=(const MATRIX& _M)
        {
            if (this == &_M)
                return *this;
            delete[] M;
            // Копирование данных M <= _M
            this->rows = _M.rows;
            this->columns = _M.columns;
            
            //выделить память
            this->M = (T*) new T[columns * rows];

            // Скопировать элементы
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < columns; j++)
                    M[j + columns * i] = _M.M[j + columns * i];
            return *this;
        }

        // оператор копирования - обязательный
        MATRIX& operator=(MATRIX&& _M)
        {
            if (!empty())
            {
                delete[] M;
            }

            // Копирование данных M <= _M
            rows = _M.rows;
            columns = _M.columns;

            // Скопировать указатель
            M = _M.M; 
            _M.M = nullptr;
            return *this;
        }

        MATRIX& operator+=(const MATRIX& _M1)
        {
            *this + _M1;
            return *this;
        }

        MATRIX operator+(const MATRIX& _M1)
        {
            if (rows == _M1.rows && columns == _M1.columns)
            {
                MATRIX tmp(rows, columns);

                T* dev_M;
                T* dev_M1;
                size_t size = rows * columns * sizeof(T);

                cudaMalloc((void**)&dev_M, size);
                cudaMalloc((void**)&dev_M1, size);
                cudaMemcpy(dev_M, this->M, size, cudaMemcpyHostToDevice);
                cudaMemcpy(dev_M1, _M1.M, size, cudaMemcpyHostToDevice);


                size_t THREADS_PER_BLOCK = 1024;
                size_t BLOCKS_PER_GRID = std::min(size_t(12),
                    (rows * columns + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

                MatrixOperatorPlus << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (dev_M, dev_M1, this->rows, this->columns);
                cudaMemcpy(tmp.M, dev_M, size, cudaMemcpyDeviceToHost);

                cudaFree(dev_M);
                cudaFree(dev_M1);
                return MATRIX(std::move(tmp));
            }
            //возвращаем матрицу
            return *this;
        }

        MATRIX& operator-=(const MATRIX& _M1)
        {
            *this - _M1;
            return *this;
        }
        MATRIX& operator-(const MATRIX& _M1)
        {
            if (rows == _M1.rows && columns == _M1.columns)
            {
                T* dev_M;
                T* dev_M1;
                size_t size = rows * columns * sizeof(T);

                cudaMalloc((void**)&dev_M, size);
                cudaMalloc((void**)&dev_M1, size);
                cudaMemcpy(dev_M, this->M, size, cudaMemcpyHostToDevice);
                cudaMemcpy(dev_M1, _M1.M, size, cudaMemcpyHostToDevice);


                size_t THREADS_PER_BLOCK = 1024;
                size_t BLOCKS_PER_GRID = std::min(size_t(12),
                    (rows * columns + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

                MatrixOperatorMinus << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (dev_M, dev_M1, this->rows, this->columns);
                cudaMemcpy(this->M, dev_M, rows * columns * sizeof(T), cudaMemcpyDeviceToHost);

                cudaFree(dev_M);
                cudaFree(dev_M1);
            }
            //возвращаем матрицу
            return *this;
        }

        MATRIX& operator*=(const MATRIX& _M1)
        {
            *this - _M1;
            return *this;
        }

        MATRIX operator*(const MATRIX& _M1)
        {
            if (columns == _M1.rows)
            {

                MATRIX tmp(rows, _M1.columns);

                T* dev_res;
                T* dev_M1;
                T* dev_M2;

                size_t sidx = columns;
                size_t newsize = rows * _M1.columns * sizeof(T);

                cudaMalloc((void**)&dev_res, newsize);
                cudaMalloc((void**)&dev_M1, rows * columns * sizeof(T));
                cudaMalloc((void**)&dev_M2, _M1.rows * _M1.columns * sizeof(T));
                cudaMemcpy(dev_M1, this->M  , rows * columns * sizeof(T),         cudaMemcpyHostToDevice);
                cudaMemcpy(dev_M2, _M1.M    , _M1.rows * _M1.columns * sizeof(T), cudaMemcpyHostToDevice);
                

                size_t THREADS_PER_BLOCK = 1024;
                size_t BLOCKS_PER_GRID = std::min(size_t(12),
                    (rows * columns + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

                MatrixOperatorMult <<<BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > 
                                    (dev_res,dev_M1, dev_M2, this->rows, this->columns,sidx);



                cudaMemcpy(tmp.get_ptr(), dev_res,newsize, cudaMemcpyDeviceToHost);

                cudaFree(dev_M1);
                cudaFree(dev_M2);
                cudaFree(dev_res);
                return MATRIX(std::move(tmp));
            }
            //возвращаем матрицу
            printf("Матрицы не согласованны\n");
            return *this;
        }
      

        MATRIX& Transpose()
        {
            if (!(this->empty())) 
            {
                T* dev_M,*dev_tmp;
                size_t size = rows * columns * sizeof(T);

                cudaMalloc((void**)&dev_M, size);
                cudaMalloc((void**)&dev_tmp, size);
                cudaMemcpy(dev_M, this->M, size, cudaMemcpyHostToDevice);

                size_t THREADS_PER_BLOCK = 64;
                size_t BLOCKS_PER_GRID = std::min(size_t(12),
                    (rows * columns + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

                MatrixTranspose <<< 1, THREADS_PER_BLOCK >> > (dev_tmp, dev_M, this->rows, this->columns);
                cudaMemcpy(this->M, dev_tmp,size, cudaMemcpyDeviceToHost);

                cudaFree(dev_M);
                cudaFree(dev_tmp);

                std::swap(this->rows, this->columns);
            }
            return *this;
        }

        // Деструктор - освобождает память, выделенную для матрицы
        ~MATRIX()
        {
            if (!this->empty())
                this->erase();
        }
    };

    template <typename T>
    __global__ void MatrixOperatorPlus(T* M1,  T*  M2, uint32_t rows, uint32_t columns)
    {
        size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        while (tid < rows * columns) {
            M1[tid] += M2[tid];
            tid += blockDim.x * gridDim.x;
        }
    };

    template <typename T>
    __global__ void MatrixOperatorMinus(T* M1,T*  M2, uint32_t rows, uint32_t columns)
    {
        size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        while (tid < rows * columns) {
            M1[tid] -= M2[tid];
            tid += blockDim.x * gridDim.x;
        }
    };

    template <typename T>
    __global__ void MatrixOperatorMult(T* res,T* M1, T* M2, uint32_t rows, uint32_t columns,size_t msidx)
    {
        size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        size_t j = tid % columns;
        size_t i = int(tid / columns);
        size_t sidx = 0;

        if (tid < rows * columns &&  j + columns * i < rows * columns)
            res[j + columns * i] = 0;

        while (sidx < msidx && tid < rows * columns && i < rows && j < columns ) {
            res[j + columns * i] += M1[sidx + columns * i] * M2[j + columns * sidx];
            sidx += 1;
        }
        return;
    };

    template <typename T> 
    __device__ inline void dev_swap(T& a, T& b)
    {
        T c(a); a = b; b = c;
    }

    template <typename T>
    __global__ void MatrixTranspose(T* tmp, T* M1, size_t rows, size_t columns)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int j = tid % columns;
        int i = int(tid / columns);

        if(i < rows && j < columns) 
        {
            tmp[i + rows * j] = M1[j + columns * i];
        }
        return;
    };




    void test()
    {
        // тест для класса MATRIX
        MATRIX<int> M(3, 3);
        M.Print("M");

        // Заполнить матрицу значеннями по формуле
        int i, j;
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                M.Set(i, j,  j + 3 * i);

        M.Print("M");

        MATRIX<int> M2 = M; // вызов конструктора копирования
        M2.Print("M2: вызов конструктора копирования");

        MATRIX<int> M3; // вызов оператора копирования - проверка
        M3 = M + M2;
        M3.Print("M3: вызов оператора копирования - проверка");

        MATRIX<int> M4;
        M4 = M3 = M2 = M; // вызов оператора копирования в виде "цепочки"

        M4.Print("M4: вызов оператора копирования в виде цепочки");
        M4.Transpose();
        M4.Print("M4: T");


        (M + M2).RandInsert().Print().Transpose().Print();

        M.Print();
        M2.Print();
        (M * M2).Print("mult");
    }
}