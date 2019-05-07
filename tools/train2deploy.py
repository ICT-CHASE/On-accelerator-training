"""
Convert train prototxt to deploy

DONot use this!
"""

from __future__ import print_function
import numpy
import time
import os
import re
import argparse

def train2deploy(train_prototxt, size, deploy_prototxt):
    """
    Convert a train prototxt file to deploy prototxt file by removing labels
    and adding correct input shape.
    
    :param train_prototxt: path to train prototxt
    :type train_prototxt: string
    :param size: blob input size in (batch_size, channels, height, width)
    :type size: (int, int, int, int)
    :param deploy_prototxt: path to deploy prototxt
    :type deploy_prototxt: string
    """
    
    assert len(size) == 4
    
    def replace_input_type(prototxt):
        """
        Replace the input type.
        """
        
        return prototxt.replace('type: "Data"', 'type: "Input"')
    
    def remove_labels(prototxt):
        """
        Replace the input labels.
        """
        
        return prototxt.replace('top: "labels"\n', '')
    
    def replace_input_param(prototxt, size):
        """
        Replace the input_param with a correct one in the given size.
        """
        
        started = False
        lines_old = prototxt.split('\n')
        lines_new = []
        
        for line in lines_old:
            if line.find('data_param') >= 0:
                started = True
            elif started:
                if line.find('}') >= 0:
                    lines_new.append('  input_param { shape: { dim: ' + str(size[0]) + ' dim: ' + str(size[1]) + ' dim: ' + str(size[2]) + ' dim: ' + str(size[3]) + ' } }')
                    started = False
            else:
                lines_new.append(line)
        
        return '\n'.join(lines_new)
    
    def remove_transform_param(prototxt):
        """
        Remove transform param.
        """
        
        occurences = [(m.start(), m.end()) for m in re.finditer(r'\n[ \t]*transform_param[ \t]*\{[a-zA-Z0-9,.:" \t\r\n]*\}', prototxt)]
        
        if len(occurences) > 0:
            start = occurences[-1][0]
            end = occurences[-1][1]
            return prototxt[:start] + prototxt[end:]
        else:
            return prototxt
    
    def remove_loss(prototxt):
        """
        Remove the loss layer.
        """
        
        occurences = [m.start() for m in re.finditer(r'layer[ \t]*\{[a-zA-Z:" \t\r\n]*top:[ \t]*"loss"', prototxt)]
        
        if len(occurences) > 0:
            index = occurences[-1]
            return prototxt[:index]
        else:
            return prototxt
        
    with open(train_prototxt) as train:
        with open(deploy_prototxt, 'w') as deploy:
            prototxt_old = train.read()
            prototxt_new = replace_input_type(prototxt_old)
            prototxt_new = remove_labels(prototxt_new)
            prototxt_new = replace_input_param(prototxt_new, size)
            prototxt_new = remove_transform_param(prototxt_new)
            prototxt_new = remove_loss(prototxt_new)
            deploy.write(prototxt_new)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert train prototxt to deploy')
    parser.add_argument('--train', type=str,
                        help='train prototxt', default='')
    parser.add_argument('--deploy', type=str,
                        help='deploy prototxt', default='')

    parser.add_argument('--batch_size', type=int,
                        help='batch size of input', default=1)
    parser.add_argument('--channels', type=int,
                        help='channels of input', default=3)
    parser.add_argument('--height', type=int,
                        help='height of input', default=227)
    parser.add_argument('--width', type=int,
                        help='width of input', default=227)
    args = parser.parse_args()

    print('input train prototxt: {}'.format(args.train)) 
    print('output deploy prototxt: {}'.format(args.deploy))
    train2deploy(args.train, (args.batch_size, args.channels, args.height, args.width),
                 args.deploy)

