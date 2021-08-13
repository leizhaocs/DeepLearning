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

/* delete the ith argument */
void del_arg(int argc, char **argv, int index)
{
    int i;
    for (i = index; i < argc-1; ++i)
    {
        argv[i] = argv[i+1];
    }
    argv[i] = 0;
}

/* check if an argument exists, 1: yes, 0: no */
int find_arg(int *argc, char *argv[], const char *arg)
{
    for (int i = 0; i < *argc; ++i)
    {
        if (!argv[i])
        {
            continue;
        }
        if (0 == strcmp(argv[i], arg))
        {
            del_arg(*argc, argv, i);
            (*argc)--;
            return 1;
        }
    }
    return 0;
}
