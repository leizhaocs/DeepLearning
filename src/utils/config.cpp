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

/* load network parameters from config file */
Params *build_network(const char *config, int &num_layers, int &num_epochs, int &batch_size,
                      float &lr_begin, float &lr_decay, string &loss, int &show_acc)
{
    Params *layers;

    ifstream cfg(config, ifstream::in);

    string line;

    int layer_index = 0;

    while (getline(cfg, line))
    {
        if (line[0] == '#')
        {
            continue;
        }
        else if (line[0] == '[')
        {
            int pos = line.find(']');
            string layername = line.substr(1, pos-1);

            if (layername == "global")
            {
                float lr_end;

                while (true)
                {
                    getline(cfg, line);
                    if (line[0] == '#')
                    {
                        continue;
                    }
                    int pos = line.find('=');
                    if (pos == string::npos)
                    {
                        break;
                    }

                    string key = line.substr(0, pos-1);
                    if (key == "layers")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        valuestream >> num_layers;
                    }
                    else if (key == "epochs")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        valuestream >> num_epochs;
                    }
                    else if (key == "batch_size")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        valuestream >> batch_size;
                    }
                    else if (key == "lr_begin")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        valuestream >> lr_begin;
                    }
                    else if (key == "lr_end")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        valuestream >> lr_end;
                    }
                    else if (key == "loss")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        valuestream >> loss;
                    }
                    else if (key == "show_acc")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        valuestream >> show_acc;
                    }
                }
                lr_decay = (lr_begin - lr_end) / num_epochs;
                layers = new Params[num_layers];
                continue;
            }
            else if (layername == "input")
            {
                layers[layer_index].addString("type", "input");

                while (true)
                {
                    getline(cfg, line);
                    if (line[0] == '#')
                    {
                        continue;
                    }
                    int pos = line.find('=');
                    if (pos == string::npos)
                    {
                        break;
                    }

                    string key = line.substr(0, pos-1);
                    if (key == "shape")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        int c, h, w;
                        valuestream >> c;
                        valuestream >> h;
                        valuestream >> w;
                        layers[layer_index].addVectori("shape", vector<int> {c, h, w});
                    }
                }
            }
            else if(layername == "convolution")
            {
                layers[layer_index].addString("type", "convolution");

                while (true)
                {
                    getline(cfg, line);
                    if (line[0] == '#')
                    {
                        continue;
                    }
                    int pos = line.find('=');
                    if (pos == string::npos)
                    {
                        break;
                    }

                    string key = line.substr(0, pos-1);
                    if (key == "filterSize")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        int f, h, w;
                        valuestream >> f;
                        valuestream >> h;
                        valuestream >> w;
                        layers[layer_index].addVectori("filterSize", vector<int> {f, h, w});
                    }
                    else if (key == "stride")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        int h, w;
                        valuestream >> h;
                        valuestream >> w;
                        layers[layer_index].addVectori("stride", vector<int> {h, w});
                    }
                    else if (key == "padding")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        int h, w;
                        valuestream >> h;
                        valuestream >> w;
                        layers[layer_index].addVectori("padding", vector<int> {h, w});
                    }
                }
            }
            else if(layername == "pool")
            {
                layers[layer_index].addString("type", "pool");

                while (true)
                {
                    getline(cfg, line);
                    if (line[0] == '#')
                    {
                        continue;
                    }
                    int pos = line.find('=');
                    if (pos == string::npos)
                    {
                        break;
                    }

                    string key = line.substr(0, pos-1);
                    if (key == "poolType")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        string pooltype;
                        valuestream >> pooltype;
                        layers[layer_index].addString("poolType", pooltype.c_str());
                    }
                    else if (key == "filterSize")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        int h, w;
                        valuestream >> h;
                        valuestream >> w;
                        layers[layer_index].addVectori("filterSize", vector<int> {h, w});
                    }
                    else if (key == "stride")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        int h, w;
                        valuestream >> h;
                        valuestream >> w;
                        layers[layer_index].addVectori("stride", vector<int> {h, w});
                    }
                    else if (key == "padding")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        int h, w;
                        valuestream >> h;
                        valuestream >> w;
                        layers[layer_index].addVectori("padding", vector<int> {h, w});
                    }
                }
            }
            else if(layername == "full")
            {
                layers[layer_index].addString("type", "full");

                while (true)
                {
                    getline(cfg, line);
                    if (line[0] == '#')
                    {
                        continue;
                    }
                    int pos = line.find('=');
                    if (pos == string::npos)
                    {
                        break;
                    }

                    string key = line.substr(0, pos-1);
                    if (key == "length")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        int length;
                        valuestream >> length;
                        layers[layer_index].addScalari("length", length);
                    }
                }
            }
            else if(layername == "activation")
            {
                layers[layer_index].addString("type", "activation");

                while (true)
                {
                    getline(cfg, line);
                    if (line[0] == '#')
                    {
                        continue;
                    }
                    int pos = line.find('=');
                    if (pos == string::npos)
                    {
                        break;
                    }

                    string key = line.substr(0, pos-1);
                    if (key == "nonlinear")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        string nonlinear;
                        valuestream >> nonlinear;
                        layers[layer_index].addString("nonlinear", nonlinear.c_str());
                    }
                }
            }
            else if(layername == "batchnormalization")
            {
                layers[layer_index].addString("type", "batchnormalization");

                while (true)
                {
                    getline(cfg, line);
                    if (line[0] == '#')
                    {
                        continue;
                    }
                    int pos = line.find('=');
                    if (pos == string::npos)
                    {
                        break;
                    }

                    string key = line.substr(0, pos-1);
                    if (key == "channels")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        int channels;
                        valuestream >> channels;
                        layers[layer_index].addScalari("channels", channels);
                    }
                }
            }
            else if(layername == "dropout")
            {
                layers[layer_index].addString("type", "dropout");

                while (true)
                {
                    getline(cfg, line);
                    if (line[0] == '#')
                    {
                        continue;
                    }
                    int pos = line.find('=');
                    if (pos == string::npos)
                    {
                        break;
                    }

                    string key = line.substr(0, pos-1);
                    if (key == "rate")
                    {
                        string value = line.substr(pos+1, string::npos);
                        stringstream valuestream(value);
                        float rate;
                        valuestream >> rate;
                        layers[layer_index].addScalarf("rate", rate);
                    }
                }
            }
            layer_index++;
            if (layer_index >= num_layers)
            {
                break;
            }
        }
    }

    cfg.close();

    return layers;
}
