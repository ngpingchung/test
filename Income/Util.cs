using System;

namespace Income
{

    static class TransposeRowsColumnsExtension
    {
        public static double[][] TransposeRowsAndColumns(double[][] arr)
        {
            int rowCount = arr.Length;
            int columnCount = arr[0].Length;
            double[][] transposed = new double[columnCount][];
            if (rowCount == columnCount)
            {
                transposed = (double[][])arr.Clone();
                for (int i = 1; i < rowCount; i++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        double temp = transposed[i][j];
                        transposed[i][j] = transposed[j][i];
                        transposed[j][i] = temp;
                    }
                }
            }
            else
            {
                for (int column = 0; column < columnCount; column++)
                {
                    transposed[column] = new double[rowCount];
                    for (int row = 0; row < rowCount; row++)
                    {
                        transposed[column][row] = arr[row][column];
                    }
                }
            }
            return transposed;
        }

    }

}