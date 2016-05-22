using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNet
{
    class Program
    {
        static void Main(string[] args)
        {
            var inputA = new DenseMatrix(4, 4);
            for (int row = 0; row < inputA.RowCount; row++)
            {
                for (int column = 0; column < inputA.ColumnCount; column++)
                {
                    inputA[row, column] = column / 4f;
                }
            }

            // we convert each kernel into a column (tall 9)
            // so the convolution would have to be 'n' rows with length 9
            var conv1 = new DenseMatrix(1, 9);
            conv1[0, 1] = 0.5;
            //conv1[0, 4] = 0.5;

            for (int i = 0; i < 1000; i++)
            {
                var inputA_col = im2col(inputA);
                var features1_col = conv1 * inputA_col;
                var features1 = col2im(features1_col);
                //DenseMatrix m = new DenseMatrix(4, 4);
                // DenseMatrix m2 = new DenseMatrix()
                Console.WriteLine("inputA " + inputA);
                Console.WriteLine("conv1 " + conv1);
                Console.WriteLine("inputA_col " + inputA_col);
                Console.WriteLine("features1_col " + features1_col);
                Console.WriteLine("features1[0] " + features1[0]);
                //Console.WriteLine("features1[1] " + features1[1]);

                var current = features1_col.RowSums()[0];
                var error = 10 - current;

                Console.WriteLine("current " + current);
                Console.WriteLine("error " + error);

                Console.WriteLine("train");

                var conv1_localGradient = (DenseMatrix)inputA_col.RowSums().ToRowMatrix();
                //Console.WriteLine(DenseMatrix.Create(1, 9, 1));
                //Console.WriteLine(DenseMatrix.Create(1, 9, 1) * input);_col);
                Console.WriteLine("conv1_localGradient " + conv1_localGradient);

                broadcastAdd(conv1, conv1_localGradient * 0.001 * error);

                Console.WriteLine(col2im(conv1_localGradient)[0]);
                //for (int row = 0; row < conv1.RowCount; row++)
                //{
                //    for (int column = 0; column < conv1.ColumnCount; column++)
                //    {
                //        conv1[row, column] += conv1_localGradient[0, column] * 0.001 * error;
                //    }
                //}
                Console.ReadLine();
            }
        }

        static void broadcastAdd(DenseMatrix m, DenseMatrix add)
        {
            for (int row = 0; row < m.RowCount; row++)
            {
                for (int column = 0; column < m.ColumnCount; column++)
                {
                    m[row, column] += add[0, column];
                }
            }
        }

        static DenseMatrix im2col(DenseMatrix image)
        {
            var r = new DenseMatrix(9, image.RowCount * image.ColumnCount);

            for (int column = 0; column < r.ColumnCount; column++)
            {
                var imageRow = column / image.ColumnCount;
                var imageColumn = column % image.ColumnCount;

                r[0, column] = getValue(image, imageRow - 1, imageColumn - 1);
                r[1, column] = getValue(image, imageRow - 1, imageColumn + 0);
                r[2, column] = getValue(image, imageRow - 1, imageColumn + 1);
                r[3, column] = getValue(image, imageRow + 0, imageColumn - 1);
                r[4, column] = getValue(image, imageRow + 0, imageColumn + 0);
                r[5, column] = getValue(image, imageRow + 0, imageColumn + 1);
                r[6, column] = getValue(image, imageRow + 1, imageColumn - 1);
                r[7, column] = getValue(image, imageRow + 1, imageColumn + 0);
                r[8, column] = getValue(image, imageRow + 1, imageColumn + 1);
            }

            return r;
        }
        static DenseMatrix[] col2im(DenseMatrix features)
        {
            var returnValue = new DenseMatrix[features.RowCount];
            for (int featureIndex = 0; featureIndex < features.RowCount; featureIndex++)
            {
                var s = (int)(Math.Sqrt(features.ColumnCount) + 0.0001);
                var r = new DenseMatrix(s);

                for (int i = 0; i < features.ColumnCount; i++)
                {
                    var row = i / s;
                    var column = i % s;

                    r[row, column] = features[featureIndex, i];
                }
                returnValue[featureIndex] = r;
            }
            return returnValue;
        }

        static double getValue(DenseMatrix m, int r, int c)
        {
            var returnValue = 0.0;
            if (0 <= r && r < m.RowCount && 0 <= c && c < m.ColumnCount)
            {
                returnValue = m[r, c];
            }
            return returnValue;
        }
    }
}
