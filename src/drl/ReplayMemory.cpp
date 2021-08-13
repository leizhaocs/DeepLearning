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

/* constructor */
ReplayMemory::ReplayMemory(int capacity, int n, int h, int w)
{
    capacity_ = capacity;
    begin_ = 0;
    end_ = 0;
    size_ = 0;
    mem_ = new Transition*[capacity_];
    for (int i = 0; i < capacity_; i++)
    {
        mem_[i] = new Transition(n, h, w, 0);
    }

    num_frames_ = n + 1;
    cur_frame_ = num_frames_ - 1;
    frames_ = new float*[num_frames_];
    for (int i = 0; i < num_frames_; i++)
    {
        frames_[i] = new float[h*w];
        for (int j = 0; j < h*w; j++)
        {
            frames_[i][j] = 0;
        }
    }
    h_ = h;
    w_ = w;
}

/* destructor */
ReplayMemory::~ReplayMemory()
{
    for (int i = 0; i < capacity_; i++)
    {
        delete mem_[i];
    }
    delete [] mem_;
    for (int i = 0; i < num_frames_; i++)
    {
        delete [] frames_[i];
    }
    delete [] frames_;
}

/* reset at the starting of each new episode */
void ReplayMemory::reset()
{
    cur_frame_ = num_frames_ - 1;
    for (int i = 0; i < num_frames_; i++)
    {
        for (int j = 0; j < h_*w_; j++)
        {
            frames_[i][j] = 0;
        }
    }
}

/* get number of transitions in replay memory */
int ReplayMemory::getSize()
{
    return size_;
}

/* add a new transition into replay memory */
void ReplayMemory::addTransition(float r, int a, int final_state)
{
    Transition *trans;
    if (size_ == 0)
    {
        trans = mem_[end_];
        size_++;
    }
    else
    {
        end_ = (end_ + 1) % capacity_;
        if (end_ == begin_)
        {
             begin_ = (begin_ + 1) % capacity_;
        }
        else
        {
            size_++;
        }
        trans = mem_[end_];
    }

    trans->getReward() = r;
    trans->getAction()->setAction(a);
    Observation *curOb = trans->getCurrent();
    Observation *nextOb = trans->getNext();
    for (int i = curOb->getLength()-1, f = cur_frame_; i >= 0; i--)
    {
        int f_ = f - 1;
        if (f_ < 0)
        {
            f_ = num_frames_ - 1;
        }

        nextOb->setImage(i, frames_[f]);
        curOb->setImage(i, frames_[f_]);

        f = f_;
    }
    trans->get_final_state() = final_state;
}

/* get the latest transition */
Transition *ReplayMemory::getLatestTran()
{
    return mem_[end_];
}

/* randomly sample a batch of transitions, without removing them from memory */
bool ReplayMemory::sampleTransitions(int sampleSize, Transition **sample)
{
    if (size_ < sampleSize)
    {
        return false;
    }

    vector<int> selected;
    for (int i = 0; i < sampleSize; i++)
    {
        while (1)
        {
            int index = rand() % size_;
            index = (begin_ + index) % capacity_;
            bool found = false;
            for (int j = 0; j < selected.size(); j++)
            {
                if (selected[j] == index)
                {
                    found = true;
                    break;
                }
            }
            if (found == false)
            {
                selected.push_back(index);
                sample[i] = mem_[index];
                break;
            }
        }
    }

    return true;
}

/* get the frame buffer to write into */
float *ReplayMemory::getFrameBuffer()
{
    cur_frame_ = (cur_frame_ + 1) % num_frames_;
    return frames_[cur_frame_];
}
