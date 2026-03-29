#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <windows.h>

using namespace std;

// 矩阵列向量内积
void matrix_dot_naive(const vector<vector<double>>& mat, const vector<double>& vec, vector<double>& sum, int n) {
    for (int j = 0; j < n; ++j) {
        sum[j] = 0.0;
        for (int i = 0; i < n; ++i) {
            sum[j] += mat[i][j] * vec[i];
        }
    }
}

void matrix_dot_cache_optimized(const vector<vector<double>>& mat, const vector<double>& vec, vector<double>& sum, int n) {
    for (int j = 0; j < n; ++j) {
        sum[j] = 0.0;
    }

    for (int i = 0; i < n; ++i) {
        // 需要用到 i 的数据在第一层循环优先读取
        double x = vec[i];
        const vector<double>& row = mat[i];
        for (int j = 0; j < n; ++j) {
            sum[j] += row[j] * x;
        }
    }
}

void matrix_dot_cache_unroll4(const vector<vector<double>>& mat, const vector<double>& vec, vector<double>& sum, int n) {
    for (int j = 0; j < n; ++j) {
        sum[j] = 0.0;
    }

    for (int i = 0; i < n; ++i) {
        double x = vec[i];
        const vector<double>& row = mat[i];
        int j = 0;
        for (; j + 3 < n; j += 4) {
            sum[j] += row[j] * x;
            sum[j + 1] += row[j + 1] * x;
            sum[j + 2] += row[j + 2] * x;
            sum[j + 3] += row[j + 3] * x;
        }
        for (; j < n; ++j) {
            sum[j] += row[j] * x;
        }
    }
}

void matrix_dot_cache_unroll8(const vector<vector<double>>& mat, const vector<double>& vec, vector<double>& sum, int n) {
    for (int j = 0; j < n; ++j) {
        sum[j] = 0.0;
    }

    for (int i = 0; i < n; ++i) {
        double x = vec[i];
        const vector<double>& row = mat[i];
        int j = 0;
        for (; j + 7 < n; j += 8) {
            sum[j] += row[j] * x;
            sum[j + 1] += row[j + 1] * x;
            sum[j + 2] += row[j + 2] * x;
            sum[j + 3] += row[j + 3] * x;
            sum[j + 4] += row[j + 4] * x;
            sum[j + 5] += row[j + 5] * x;
            sum[j + 6] += row[j + 6] * x;
            sum[j + 7] += row[j + 7] * x;
        }
        for (; j < n; ++j) {
            sum[j] += row[j] * x;
        }
    }
}

// 初始化
void init_exp1(vector<vector<double>>& mat, vector<double>& vec, int n) {
    for (int i = 0; i < n; ++i) {
        vec[i] = i;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = i + j;
        }
    }
}

// 正确性校验
bool correct_check(const vector<double>& a, const vector<double>& b, double eps = 1e-8) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (fabs(a[i] - b[i]) > eps) {
            return false;
        }
    }
    return true;
}

int main() {
    cout << fixed << setprecision(3);
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    vector<int> matrix_sizes = { 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120 };

    for (int n : matrix_sizes) {
        vector<vector<double>> mat(n, vector<double>(n));
        vector<double> vec(n);
        vector<double> result_naive(n), result_cache(n), result_unroll4(n), result_unroll8(n);
        init_exp1(mat, vec, n);

        // 正确性验证
        matrix_dot_naive(mat, vec, result_naive, n);
        matrix_dot_cache_optimized(mat, vec, result_cache, n);
        matrix_dot_cache_unroll4(mat, vec, result_unroll4, n);
        matrix_dot_cache_unroll8(mat, vec, result_unroll8, n);
        bool correct1 = correct_check(result_naive, result_cache);
        bool correct2 = correct_check(result_naive, result_unroll4);
        bool correct3 = correct_check(result_naive, result_unroll8);

        int repeat = 1;
        if (n <= 512) repeat = 200;
        else if (n <= 1024) repeat = 50;
        else if (n <= 2048) repeat = 20;
        else repeat = 10;

        QueryPerformanceCounter(&start);
        for (int i = 0; i < repeat; ++i)
            matrix_dot_naive(mat, vec, result_naive, n);
        QueryPerformanceCounter(&end);
        double t_naive = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
        double t_naive_avg = t_naive / repeat;

        QueryPerformanceCounter(&start);
        for (int i = 0; i < repeat; ++i)
            matrix_dot_cache_optimized(mat, vec, result_cache, n);
        QueryPerformanceCounter(&end);
        double t_cache = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
        double t_cache_avg = t_cache / repeat;

        QueryPerformanceCounter(&start);
        for (int i = 0; i < repeat; ++i)
            matrix_dot_cache_unroll4(mat, vec, result_unroll4, n);
        QueryPerformanceCounter(&end);
        double t_unroll4 = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
        double t_unroll4_avg = t_unroll4 / repeat;

        QueryPerformanceCounter(&start);
        for (int i = 0; i < repeat; ++i)
            matrix_dot_cache_unroll8(mat, vec, result_unroll8, n);
        QueryPerformanceCounter(&end);
        double t_unroll8 = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
        double t_unroll8_avg = t_unroll8 / repeat;

        cout << "n = " << n << "\n";
        cout << "  正确性检验:" << ((correct1 && correct2 && correct3) ? "通过" : "未通过") << "\n";
        cout << "  naive总时间      : " << t_naive << " ms\n";
        cout << "  naive平均时间    : " << t_naive_avg << " ms\n";
        cout << "  cache优化总时间  : " << t_cache << " ms\n";
        cout << "  cache平均时间    : " << t_cache_avg << " ms\n";
        cout << "  cache优化加速比  : " << (t_naive / t_cache) << "\n";
        cout << "  unroll4总时间    : " << t_unroll4 << " ms\n";
        cout << "  unroll4平均时间  : " << t_unroll4_avg << " ms\n";
        cout << "  unroll4加速比    : " << (t_naive / t_unroll4) << "\n";
        cout << "  unroll8总时间    : " << t_unroll8 << " ms\n";
        cout << "  unroll8平均时间  : " << t_unroll8_avg << " ms\n";
        cout << "  unroll8加速比    : " << (t_naive / t_unroll8) << "\n";
        cout << "--------------------------------------\n";
    }

    return 0;
}