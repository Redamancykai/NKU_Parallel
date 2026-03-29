#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <windows.h>

using namespace std;

// 数组求和
double sum_naive(const vector<double>& arr) {
    double sum = 0.0;
    for (size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}

double sum_two_way(const vector<double>& arr) {
    double sum1 = 0.0, sum2 = 0.0;
    size_t i = 0;
    size_t n = arr.size();

    for (; i + 1 < n; i += 2) {
        sum1 += arr[i];
        sum2 += arr[i + 1];
    }
    if (i < n) sum1 += arr[i];

    return sum1 + sum2;
}

double sum_four_way(const vector<double>& arr) {
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    size_t i = 0;
    size_t n = arr.size();

    for (; i + 3 < n; i += 4) {
        sum0 += arr[i];
        sum1 += arr[i + 1];
        sum2 += arr[i + 2];
        sum3 += arr[i + 3];
    }

    for (; i < n; ++i) {
        sum0 += arr[i];
    }

    return sum0 + sum1 + sum2 + sum3;
}

// 递归
double sum_recursive_func(const double* data, int left, int right) {
    if (left == right) return data[left];
    if (right - left == 1) return data[left] + data[right];

    int mid = left + (right - left) / 2;
    double s1 = sum_recursive_func(data, left, mid);
    double s2 = sum_recursive_func(data, mid + 1, right);
    return s1 + s2;
}

double sum_recursive(const vector<double>& arr) {
    if (arr.empty()) return 0.0;
    return sum_recursive_func(arr.data(), 0, arr.size() - 1);
}

// 二重循环迭代
double sum_iterative_func(vector<double>& arr) {
    if (arr.empty()) return 0.0;

    int n = arr.size();
    for (int m = n; m > 1; m /= 2) {
        for (int i = 0; i < m / 2; ++i) {
            arr[i] = arr[2 * i] + arr[2 * i + 1];
        }
        arr.resize(m / 2);
    }
    return arr[0];
}

double sum_iterative(const vector<double>& arr) {
    if (arr.empty()) return 0.0;
    vector<double> temp = arr;
    return sum_iterative_func(temp);
}

// 初始化
void init_exp2(vector<double>& arr) {
    for (size_t i = 0; i < arr.size(); ++i) {
        arr[i] = i;
    }
}

int main() {
    cout << fixed << setprecision(3);
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    vector<int> sum_sizes = { 1 << 10,1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20,1 << 22, 1 << 24, 1 << 26 };

    for (int n : sum_sizes) {
        vector<double> arr(n);
        init_exp2(arr);

        double sum1 = sum_naive(arr);
        double sum2 = sum_two_way(arr);
        double sum3 = sum_four_way(arr);
        double sum4 = sum_recursive(arr);
        double sum5 = sum_iterative(arr);
        bool correct1 = fabs(sum1 - sum2) < 1e-8;
        bool correct2 = fabs(sum1 - sum3) < 1e-8;
        bool correct3 = fabs(sum1 - sum4) < 1e-8;
        bool correct4 = fabs(sum1 - sum5) < 1e-8;

        int repeat = 1;
        if (n <= 10000000) repeat = 50;
        else if (n <= 50000000) repeat = 20;
        else repeat = 10;

        QueryPerformanceCounter(&start);
        for (int i = 0; i < repeat; ++i) {
            volatile double s = sum_naive(arr);
            (void)s;
        }
        QueryPerformanceCounter(&end);
        double t_naive = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
        double t_naive_avg = t_naive / repeat;

        QueryPerformanceCounter(&start);
        for (int i = 0; i < repeat; ++i) {
            volatile double s = sum_two_way(arr);
            (void)s;
        }
        QueryPerformanceCounter(&end);
        double t_two = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
        double t_two_avg = t_two / repeat;

        QueryPerformanceCounter(&start);
        for (int i = 0; i < repeat; ++i) {
            volatile double s = sum_four_way(arr);
            (void)s;
        }
        QueryPerformanceCounter(&end);
        double t_four = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
        double t_four_avg = t_four / repeat;

        QueryPerformanceCounter(&start);
        for (int i = 0; i < repeat; ++i) {
            volatile double s = sum_recursive(arr);
            (void)s;
        }
        QueryPerformanceCounter(&end);
        double t_rec = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
        double t_rec_avg = t_rec / repeat;

        QueryPerformanceCounter(&start);
        for (int i = 0; i < repeat; ++i) {
            volatile double s = sum_iterative(arr);
            (void)s;
        }
        QueryPerformanceCounter(&end);
        double t_it = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
        double t_it_avg = t_it / repeat;

        cout << "n = " << n << "\n";
        cout << "  正确性检验:" << ((correct1 && correct2 && correct3 && correct4) ? "通过" : "未通过") << "\n";
        cout << "  naive总时间      : " << t_naive << " ms\n";
        cout << "  naive平均时间    : " << t_naive_avg << " ms\n";
        cout << "  two-way总时间    : " << t_two << " ms\n";
        cout << "  two_way平均时间  : " << t_two_avg << " ms\n";
        cout << "  two-way加速比    : " << (t_naive / t_two) << "\n";
        cout << "  four-way总时间   : " << t_four << " ms\n";
        cout << "  four_way平均时间 : " << t_four_avg << " ms\n";
        cout << "  four-way加速比   : " << (t_naive / t_four) << "\n";
        cout << "  recursive总时间  : " << t_rec << " ms\n";
        cout << "  recursive平均时间: " << t_rec_avg << " ms\n";
        cout << "  recursive加速比  : " << (t_naive / t_rec) << "\n";
        cout << "  iterative总时间  : " << t_it << " ms\n";
        cout << "  iterative平均时间: " << t_it_avg << " ms\n";
        cout << "  iterative加速比  : " << (t_naive / t_it) << "\n";
        cout << "--------------------------------------\n";
    }

    return 0;
}