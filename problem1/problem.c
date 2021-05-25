
#include <stdio.h>
#include <math.h>

int main(void)

{

  FILE *fp;             /***    ファイル保存に用いる変数  ***/
  int i, j;             /***           ループ用変数       ***/

  double  delta;        /***          時間刻み幅          ***/
  double  y, ynew;      /***           求める関数         ***/
  float  error, sdelta; /*** 変数　error と sdelta だけは単精度変数としておく．
 　                      これは gnuplot は単精度の値をプロットするためである．****/
  int num[3];           /***          ステップ数          ***/


/***            ステップ数を設定する.          ***/
  num[0] = 1000;
  num[1] = 100;
  num[2] = 10;

/***       結果を書き込むファイルを開く.       ***/
  fp = fopen("error.txt","w");

/***  3種類のステップ数の場合をループで回す.   ***/
  for (i = 0; i < 3; i++)
  {
      /***      時間ステップと初期条件の設定              ***/
       delta = 1.0/(double)num[i];
       y = 1.0;

      /***      Euler 法により x = 1 まで時間発展         ***/
       for (j = 1; j <= num[i]; j++)
       {
           ynew = y + delta * exp(-y);
           y = ynew;
       }

      /***   delta は倍精度変数なので値を単精度に直す．   ***/
       sdelta = delta;

      /***   x = 1 における誤差を単精度変数 error に格納  ***/
       error = (float)( y - log( 1.0 + exp(1.0) ) );


      /***   結果をファイルに書き込む.                    ***/
       fprintf(fp, "%f  %f\n", sdelta, error);

   }
   /*  End of Loop */

  /***      書き込んだファイルを閉じる.      ***/
  fclose(fp);

}
/*  End of Program  */