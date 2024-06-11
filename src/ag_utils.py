import torch
import re

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def remove_word(self, word):
        if word in self.word2idx:
            del self.word2idx[word]
            del self.idx2word[self.idx]
            self.idx -= 1
    
    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, node_dict):
        self.dictionary = Dictionary()
        self.num_tokens = 0
        self.node_dict = node_dict
        self.save_data(node_dict)

    def save_data(self, node_dict):
        # Save words to the dictionary
        tokens = 0
        for id, node in node_dict.items():
            words = [node['predicate']] + node['attributes']
            tokens += len(words)
            for word in words: 
                self.dictionary.add_word(word)  
        
        self.num_tokens = tokens
    
    def get_node_features(self):
        # Tokenize the node_dict content
        node_featues = torch.zeros(len(self.node_dict), len(self.dictionary))

        for idx, node in self.node_dict.items():
            words = [node['predicate']] + node['attributes']
            for word in words:
                node_featues[idx][self.dictionary.word2idx[word]] = 1

        return node_featues
    
    def get_node_types(self):
        node_types = []
        for idx, node in self.node_dict.items():
            node_types.append(node['shape'])
        return node_types
    
    def get_action_nodes(self):
        action_nodes_dict = {}
        for idx, node in self.node_dict.items():
            if node['shape'] == 'diamond':
                action_nodes_dict[idx] = node
        return action_nodes_dict
    
    def get_num_tokens(self):
        return self.num_tokens

def parse_ag_file(attack_graph_path):
    """
    Parse the attack graph file and return the nodes, edges and node properties
    """
    with open(attack_graph_path, 'r') as file:
        dot_contents = file.read()

    node_pattern = r'(\w+)\s*\[.*?\];'
    edge_pattern = r'(\w+)\s*->\s*(\w+).*?;'
    node_properties_pattern = r'\[(.+)\]'

    # find nodes, edges, and node properties
    nodes = re.findall(node_pattern, dot_contents)
    edges = re.findall(edge_pattern, dot_contents)
    node_properties = re.findall(node_properties_pattern, dot_contents)

    return nodes, edges, node_properties

def parse_node_properties(nodes, node_properties):
    """
    Parse the node properties and return a dictionary of node properties
    """
    node_dict = {}
    for item in node_properties:

        label_match = re.search('label="(.*)"', item)
        property_list = label_match.group(1).split(':')
        node_id = nodes.index(property_list[0])
        node_prop = property_list[1]
        node_compromise_prob =  float(property_list[2])

        pattern = r'(.+)\((.*)\)'
        resp = re.findall(pattern, node_prop)
        predicate = resp[0][0].strip()
        attr = resp[0][1].strip()
        
        if ',' in attr:
            attributes = attr.split(',')
        else:
            attributes = attr.split()

        for i in range(len(attributes)):
            if "'" in attributes[i]:
                attributes[i] = attributes[i].strip("'")
            elif '"' in attributes[i]:
                attributes[i] = attributes[i].strip('"')

        node_dict[node_id] = {'predicate': predicate, 'attributes': attributes, 'possibility': node_compromise_prob}

        shape_match = re.search('shape=(.*)', item)
        node_shape = shape_match.group(1)
        node_dict[node_id]['shape'] = node_shape

    return node_dict