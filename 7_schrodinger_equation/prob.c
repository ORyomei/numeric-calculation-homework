/***  課題７：  Schroedinger 方程式の固有値問題（空間１次元） ***/
/***            第二固有値（固有関数は奇関数）の計算          ***/

#include <stdio.h>
#include <math.h>

#define N 500
#define NARRAY 2 * N + 1

/***  文関数の定義  ***/
#define F(x) (-3 * (pow(1 / cosh((x)), 2)) - e)

int main(void)
{
    FILE *fp;
    double eigenf[NARRAY], xc[NARRAY];
    double x, xstart, xend, dx;
    double e;
    double u, umax, p, unew, pnew;
    double ak1u, ak2u, ak3u, ak4u;
    double ak1p, ak2p, ak3p, ak4p;
    double uprod, uprev;
    int i, j, igraph;

    /*** xの範囲指定  ***/
    xstart = -20.0;
    xend = 0;

    /*** xの刻み ***/
    dx = (xend - xstart) / N;

    /*** ファイルの指定  ***/
    fp = fopen("oddf.txt", "w");

    /*** エネルギー固有値についての繰返し  ***/
    for (j = 0; j < 5000; j++)
    {

        e = -0.093 + 0.000001 * j;

        /***  xのスタート位置  ***/
        x = xstart;

        /*** スタート位置における u と p（u の微分）の値  ***/
        u = 1.0e-5;
        p = u * sqrt(F(xstart));

        /***  配列への格納  ***/
        xc[0] = x;
        eigenf[0] = u;

        /***  u の値を最大値１に規格化するため，u の最大値を求める準備  ***/
        umax = 0;

        /***  4次の Runge-Kutta 法で，xstart から xend まで解く  ***/
        for (i = 1; i <= N; i++)
        {
            ak1u = p;
            ak1p = F(x) * u;

            ak2u = p + ak1p * dx / 2;
            ak2p = F(x + dx / 2) * (u + ak1u * dx / 2);

            ak3u = p + ak2p * dx / 2;
            ak3p = F(x + dx / 2) * (u + ak2u * dx / 2);

            ak4u = p + ak3p * dx;
            ak4p = F(x + dx) * (u + ak3u * dx);

            unew = u + (ak1u + 2 * ak2u + 2 * ak3u + ak4u) * dx / 6;
            pnew = p + (ak1p + 2 * ak2p + 2 * ak3p + ak4p) * dx / 6;

            u = unew;
            p = pnew;
            x = x + dx;

            xc[i] = x;
            eigenf[i] = u;

            if (umax < fabs(u))
            {
                umax = fabs(u);
            }
        }

        /***  xend(=0)において u の値がゼロとなる点を求める  ***/
        if (j > 0)
        {
            uprod = uprev * u;
            if (uprod < 0.0)
            {
                break;
            }
        }
        uprev = u;
    }
    // printf("%e", umax);
    /***  求められたエネルギー固有値の値のディスプレイへの書き出し  ***/
    printf("energy = %12.5e\n", e);

    /***  xの正の部分へ奇関数として u を延長  ***/
    for (i = N + 1; i <= 2 * N; i++)
    {
        xc[i] = -xc[2 * N - i];
        eigenf[i] = -eigenf[2 * N - i];
    }

    /***  u の規格化（最大値＝1）  ***/
    for (i = 0; i <= 2 * N; i++)
    {
        eigenf[i] = eigenf[i] / umax;
    }

    /***  固有関数のファイルへの書き出し  ***/
    igraph = 2 * N / 200;
    for (i = 0; i <= 2 * N; i += igraph)
    {
        fprintf(fp, "%12.5e   %12.5e\n", xc[i], eigenf[i]);
    }
}