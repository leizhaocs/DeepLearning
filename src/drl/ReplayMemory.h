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

#ifndef _REPLAYMEMORY_H_
#define _REPLAYMEMORY_H_

#include "includes.h"

/* replay memory */
class ReplayMemory
{
public:
    /* constructor */
    ReplayMemory(int capacity, int n, int h, int w);

    /* destructor */
    ~ReplayMemory();

    /* reset at the starting of each new episode */
    void reset();

    /* get number of transitions in replay memory */
    int getSize();

    /* add a new transition into replay memory */
    void addTransition(float r, int a, int final_state);

    /* get the latest transition */
    Transition *getLatestTran();

    /* randomly sample a batch of transitions, without removing them from memory */
    bool sampleTransitions(int sampleSize, Transition **sample);

    /* get the frame buffer to write into */
    float *getFrameBuffer();

private:
    int capacity_;              // capacity of the memory
    int begin_;                 // the oldest transition's index in memory
    int end_;                   // the latest transition's index in memory
    int size_;                  // current number of transitions in memory
    Transition **mem_;          // transitions in the memory

    int num_frames_;            // size of frame buffer
    int cur_frame_;             // index of the latest frame in frame buffer
    float **frames_;            // buffer of consecutive frames
    int h_;                     // height of a frame
    int w_;                     // width of a frame
};

#endif
