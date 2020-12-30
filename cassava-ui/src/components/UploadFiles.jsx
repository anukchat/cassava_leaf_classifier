import React, { Component } from 'react'
import Button from '@material-ui/core/Button'
import Form from '@material-ui/core/FormControl'
import Input from '@material-ui/core/Input'
import ImageComponent from './ImageComponent'
import Typography from '@material-ui/core/Typography'
import Container from '@material-ui/core/Container';
import Box from '@material-ui/core/Box';
import axios from 'axios'

const PredictApiUrl='http://localhost:8000/predict';

export default class UploadFiles extends Component {
    constructor(props){
        super(props);
        this.state = {
            file: '',
            imagePreviewURL:'',
            likelyClass:''
          };
        // this.classes=useStyles();
        this.handlesubmit=this.handlesubmit.bind(this);
        this.handleImageChange=this.handleImageChange.bind(this);
        
    }
    
    handlesubmit(e){
        e.preventDefault();

        var formData= new FormData()
        formData.append("file",this.state.file)

        axios.post(PredictApiUrl,formData,{
            headers: {
                'Content-Type': 'multipart/form-data',
              }
        })
        .then(res => {
            console.log(JSON.stringify(res.data.likely_class));
            // console.log(res.data);
            this.setState({
                likelyClass:JSON.stringify(res.data.likely_class)
            })
          })
        .catch(error=>{
            console.log('ERROR',error)
        })
    }

    handleImageChange(e){
        e.preventDefault();

        let reader=new FileReader();
        let file=e.target.files[0];

        reader.onloadend=()=>{
            this.setState({
                file:file,
                imagePreviewURL:reader.result,
                likelyClass:''
            });
        }

        reader.readAsDataURL(file)
    }
    

    render() {
        
        let {imagePreviewURL}=this.state;
        let $imagePreview=null;
        if( imagePreviewURL){
            $imagePreview=(<ImageComponent imageUrl={imagePreviewURL}/>);
            // ()

        }
        return (
            <Container maxWidth="xs" >
                <Form onSubmit={this.handlesubmit}>
                    <Box m={2} ml={6}>
                        <Input type="file" onChange={this.handleImageChange} />
                    </Box>
                    <Box m={2} ml={6}>
                        {$imagePreview}
                    </Box>
                    <Box mt={2} ml={14} mb={2}>
                        <Button variant="contained" color='primary' onClick={this.handlesubmit} >Identify Disease</Button>
                    </Box>
                    <Typography component="div">
                        <Box textAlign="center" ml={2} mt={2}>
                            <h3>{this.state.likelyClass.replace(/["']/g, "")}</h3>    
                        </Box>        
                    </Typography>
                </Form>
            </Container>
        )
    }
}
