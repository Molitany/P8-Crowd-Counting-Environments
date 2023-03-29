import React,{useEffect,useState} from 'react'
import {View, Image, StyleSheet,Text,Button, } from 'react-native';




const DisplayImage = () => {
    const [serverMessages,setServerMessages] = useState([String]);
    const [connectedToServer,setConnectedToserver] = useState(false) ;
    const [serverState, setServerState] = useState('Loading...');
    const [imageReceived,setImageReceived] = useState(false)
    const [image, setImage] = useState(String);
    const ws = React.useRef(new WebSocket('ws://localhost:8000/density')).current;
    useEffect(() =>{
            let serverMessagesList = [String];
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
                receiveMessage(js);
                serverMessagesList.push(e.data);
                setServerMessages([...serverMessagesList]);
            };
        },[]
    
    )
    const asyncSetImage = (js:{image:string}): Promise<string> =>{
        return new Promise((resolve, reject)=>{
            const PNG = js.image.replace(/\s/g, '');
            resolve(PNG)
        }) 
    }



    const receiveMessage = (js : {image:string}) => {
        asyncSetImage(js).then((data:string ) =>{
            setImage(data)

        });
        console.log(image);
        setImageReceived(true);
    }

    const sendToSocket = () => {
        ws.send("Hello World");
        
    }
    
    function ImageToDisplay() {
        if (imageReceived) {
            return (
                <div>
                    <View>
                        <Image source={{ uri: `data:image/jpeg;base64,${image}` }} style={{ width: 300, height: 400 }} />
                    </View>

                </div>
            )
        } else {

            return (
                <div>
                    <Button
                        onPress={sendToSocket}
                        title="Send to server"
                        color="#241584"
                        accessibilityLabel="Learn more about this purple button"
                    />
                    <Image
                        style={styles.picture}
                        source={require('../assets/images/testPictureDDD.png')} />
                </div>
            )
        }
    }


    return (
        <View style={styles.container}>
            {connectedToServer ? (
                <ImageToDisplay/>
                ) : (
                <Text style = {{color:'#fff'}}>not connected</Text>
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
        width: 400,
        height: 400,
    }
})
