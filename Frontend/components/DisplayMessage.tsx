import React,{useEffect,useState} from 'react';
import { StyleSheet, View ,Text,FlatList} from 'react-native';

import message from '../assets/tester.json'

type Density = {
    zone: string;
    level: string;
    count: string;
};


const DisplayMessage = () =>{
    const [isLoading,setLoading] = useState(true);
    const [data, setData] = useState<Density[]>([]);
    
    const getMessage = async () => {
        const json = message
        try {
        setData(json.density)
        
        } catch (error) {
        console.error(error);
        } finally {
            setLoading(false);
        }
    }
    useEffect(()=> {
        getMessage();
    },[])
    
    
    return (
        <View >
            <Text style={{ color: '#fff' }}>Hello world!</Text>
            <View style = {{flex: 1, padding:24}}>
                {isLoading ? <Text style={{ color: '#fff' }}> Loading...</Text>: 
                (
                    <FlatList 
                        data={data}
                        renderItem= {({item}) => <Text style={{ color: '#fff' }}>{item.count}</Text>} />
                )}
            </View>
        </View>
    )
            
}
export default DisplayMessage;