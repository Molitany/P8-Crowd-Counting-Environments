import React,{useEffect,useState} from 'react';
import { StyleSheet, View ,Text} from 'react-native';

import message from '../assets/tester.json'

type Density = {
    zone: string;
    level: string;
    count: string;
};

const [data, setData] = useState<Density[]>([]);

const displayMessage = () =>{
    const json = message
    setData(json.density)

            
}