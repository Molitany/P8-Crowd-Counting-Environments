import React,{useEffect,useState} from 'react'
import {View, Image, StyleSheet,Text} from 'react-native';

const [serverMessages,setServerMessages] = useState([String])
const [connectedToServer,setConnectedToserver] = useState(false) 
const DisplayImage = () => {
    var ws = React.useRef(new WebSocket('ws://localhost/8000/density')).current;
    useEffect(() =>{
            let serverMessagesList = [String];
            ws.onopen = () => {
                setConnectedToserver(true);
            }
            ws.onclose = (e) => {
                setConnectedToserver(false);
            }
            ws.onmessage = (e) => {
                serverMessagesList.push(e.data);
                setServerMessages([...serverMessagesList])
            };
        }
    
    
    
    )

    return (
        <View style={styles.container}>
            {connectedToServer ? (<Image 
                style= {styles.picture}
                source={require('../assets/images/testPictureDDD.png')}/>
                ) : (
                <Text style = {{color:'#fff'}}>not connected</Text>)
            }
        </View>
    )
}

export default DisplayImage


const styles = StyleSheet.create({
    container: {
        paddingTop: 50,
    },
    picture: {
        width: 400,
        height: 400,
    }
})
