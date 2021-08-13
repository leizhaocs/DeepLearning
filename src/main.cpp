/**********************************************************************
 *
 * Copyright Lei Zhao.
 * contact: leizhao0403@gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **********************************************************************/

#include "includes.h"

bool use_gpu;

int main_dnn(int argc, char *argv[]);
int main_drl(int argc, char *argv[]);

/* main function */
int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        cout<<"Usage:"<<endl;
        cout<<"  DNN:"<<endl;
        cout<<"    Train mode: ./nn dnn train <dataset name> <network cfg file> <load weights file[null]> <save weights file[null]>"<<endl;
        cout<<"    Test  mode: ./nn dnn test  <dataset name> <network cfg file> <load weights file>"<<endl;
        cout<<"  DRL:"<<endl;
        cout<<"    ./nn drl <network cfg file> <load weights file[null]> <save weights file[null]>"<<endl;
        exit(0);
    }

    use_gpu = !find_arg(&argc, argv, "-cpu");

    if (strcmp(argv[1], "dnn") == 0)
    {
        main_dnn(argc, argv);
    }
    else if (strcmp(argv[1], "drl") == 0)
    {
        main_drl(argc, argv);
    }
    else
    {
        cout<<"Invalide commandline arguments"<<endl;
        return -1;
    }

    return 0;
}
