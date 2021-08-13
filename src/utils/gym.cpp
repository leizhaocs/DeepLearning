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

int sock = 0;

/* send all data */
void sendall(int sock, void *data, int size)
{
    char *ptr = (char *)data;
    int sent = 0;
    int ret = 0;
    while (sent < size)
    {
        ret = send(sock, ptr+sent, size-sent, 0);
        sent += ret;
    }
}

/* recieve all data */
void recvall(int sock, void *data, int size)
{
    char *ptr = (char *)data;
    int recieved = 0;
    int ret = 0;
    while (recieved < size)
    {
        ret = read(sock, ptr+recieved, size-recieved);
        recieved += ret;
    }
}

/* connect to gym server */
int gym_connect()
{
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
	{
		printf("\n Socket creation error \n");
		return -1;
	}

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(PORT);
	if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)
	{
		printf("\nInvalid address/ Address not supported \n"); 
		return -1;
	}

	if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
	{
		printf("\nConnection Failed \n");
		return -1;
	}

    return 0;
}

/* disconnect from gym server */
void gym_disconnect()
{
    char terminate = 't';
    sendall(sock, (void *)&terminate, sizeof(char));

    close(sock);
}

/* create a gym environment */
void gym_make(string env_name)
{
    char create = 'm';
    sendall(sock, (void *)&create, sizeof(char));

    int size = env_name.length();
    sendall(sock, (void *)&size, sizeof(int));
    sendall(sock, (void *)env_name.c_str(), size*sizeof(char));
}

/* close a gym environment */
void gym_close()
{
    char close = 'c';
    sendall(sock, (void *)&close, sizeof(char));
}

/* reset a gym environment, and get the observation size */
void gym_reset()
{
    char reset = 'r';
    sendall(sock, (void *)&reset, sizeof(char));
}

/* get the dimension of image */
void gym_getImageDimension(int *channels, int *height, int *width)
{
    char get = 'g';
    sendall(sock, (void *)&get, sizeof(char));

    recvall(sock, (void *)channels, sizeof(int));
    recvall(sock, (void *)height, sizeof(int));
    recvall(sock, (void *)width, sizeof(int));
}

/* render the gym */
void gym_render(float *image, int size)
{
    char render = 'p';
    sendall(sock, (void *)&render, sizeof(char));

    recvall(sock, (void *)image, size*sizeof(char));
}

/* run one step on the environment, return finished(1) or not(0) */
float gym_step_discrete(int action, int *done)
{
    char step = 'd';
    sendall(sock, (void *)&step, sizeof(char));

    sendall(sock, (void *)&action, sizeof(int));

    float reward;
    recvall(sock, (void *)&reward, sizeof(float));

    recvall(sock, (void *)done, sizeof(int));

    return reward;
}

/* run one step on the environment, return finished(1) or not(0) */
float gym_step_box(int n, float *action, int *done)
{
    char step = 'b';
    sendall(sock, (void *)&step, sizeof(char));

    sendall(sock, (void *)&n, sizeof(int));
    sendall(sock, (void *)action, sizeof(float)*n);

    float reward;
    recvall(sock, (void *)&reward, sizeof(float));

    recvall(sock, (void *)done, sizeof(int));

    return reward;
}
