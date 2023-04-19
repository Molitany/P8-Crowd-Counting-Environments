import React, { useEffect, useState,useRef } from 'react'
import { View, Image, StyleSheet, Text, Button, Dimensions} from 'react-native';
import NumberToShow from './NumberToShow';

const { width, height } = Dimensions.get('window');

const DisplayImage = () => {
    //useRef to have the webSocket asscessible in this scope.
    const websocketRef = useRef<WebSocket | null>(null);
    const [serverMessages, setServerMessages] = useState([String]);
    const [connectedToServer, setConnectedToserver] = useState(false);
    const [serverState, setServerState] = useState('Loading...');
    const [imageReceived, setImageReceived] = useState(false);
    const [image, setImage] = useState(String);
    const [count,setCount] = useState(0);
    //useEffect defining what actions happen when the events are triggered. 
    useEffect(() => {
        let serverMessagesList = [String];
        const ws = new WebSocket('ws://localhost:8000/density');

        ws.onopen = () => {
            setConnectedToserver(true);
            console.log("Connected")
        }
        ws.onclose = (e) => {
            setConnectedToserver(false);
            console.log("Lost connection")
        }
        ws.onerror = (e) => {
            setServerState(e.type);
        }
        ws.onmessage = (e) => {
            const js = JSON.parse(e.data);
            if(js.warning){
                console.log("warning");
                console.log(js.count)
                receiveWarning(js);
            }else{
                console.log(js.image)
                receiveMessage(js);
            }
            serverMessagesList.push(e.data);
            setServerMessages([...serverMessagesList]);
        };
        websocketRef.current = ws;
    }, []
    )
    /* Found out i dont actually need this promise but im keeping it just incase something later do.
    const asyncSetImage = (js: { image: string }): Promise<string> => {
        return new Promise((resolve, reject) => {
            try {
                const PNG = js.image.replace(/\s/g, '');
                resolve(PNG)
            } catch (e) {
                console.log(`Promise failed ${e}`)
                reject();
            }
        })
    }*/
    
    //Only exist to send arbitrary message to server. Likely deleted or changed later.
    const sendToSocket = () => {
        try  {
            if(websocketRef.current){
                websocketRef.current.send("Hello World");
            }}
            catch (error){
                console.log(error)
            }
        }
    
    // Is called in onmessage. Sets Image to the image string.
    const receiveMessage = (js: { image: string }) => {
        /*asyncSetImage(js).then((data: string) => {
            setImage(data)
        });*/
        const PNG = js.image.replace(/\s/g, '');
        setImage(PNG);
        setImageReceived(true);
    }
    const receiveWarning = (js: { warning: string, count:number }) => {
        /*asyncSetImage(js).then((data: string) => {
            setImage(data)
        });*/
        const PNG = js.warning.replace(/\s/g, '');
        setCount(js.count)
        setImage(PNG);
        setImageReceived(true);
    }
    const removeWarning = () => {
        setImageReceived(false)
    }
    

    function ImageToDisplay() {
        if (imageReceived) {
            return (
                <View>
                    {NumberToShow(count)}
                    <Button
                        onPress={removeWarning}
                        title="Confirm warning"
                        color="#241584"/>
                    <Image source={{ uri: `data:image/jpeg;base64,${image}` }} style={styles.picture} />
                            
                </View>
            )
        } else {

            return (
                <View>
                    <Button
                        onPress={sendToSocket}
                        title="Send to server"
                        color="#241584"
                        accessibilityLabel="Learn more about this purple button" />
                    <Image
                        style={styles.picture}
                        resizeMode="contain"

                        source={require('../assets/images/testPictureDDD.png')} />
                </View>
            )
        }
    }
    return (
        <View style={styles.container}>
            {connectedToServer ? (
                <ImageToDisplay />
            ) : (
                <Text style={{ color: '#fff' }}>not connected</Text>
            )
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
        width: width,
        height: height,
    
    }
})
