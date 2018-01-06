---
layout: post
title: "Running Jupyter on AWS"
excerpt: "A guide to setting up and running a Jupyter Notebook on an AWS EC2 gpu instance for deep learning."
date: 2018-01-02
---

## Deep Learning in the Cloud

**Amazon Web Services.** Training and evaluating deep neural networks is a computationally intensive task. Unfortunately, this problem isn't confined to deep neural networks. Natural language processing tasks can require processing large sparse word vector matrices that can take too many of a computer's resources to process in a reasonable amount of time. Indeed due to the complexity of tuning many of these processes as well as feature engineering, we may need to run many different variations of a pipeline to get adequate results. Unless you've invested in your own gpu machine learning compute rig, the only available option is to set up a remote gpu cloud compute instance. This post will cover Amazon Web Services Elastic Cloud Compute (EC2) as this was the cheaper option when I initially explored whether to go with AWS or Google Cloud.

This post consolidates several other posts on this topic, but is mostly a note to myself to remember how to do it. Getting an EC2 gpu compute instance (p2.xlarge) is fairly straightforward, but setting up Keras with the Tensorflow gpu compute backend was pretty involved since it needs to built from source and you need to transfer the Nvidia Cudo Toolkit and CUDNN libraries over to the instance.

To avoid all of this,there is a Deep Learning Amazon Machine Instance that's free to launch instances from, which has nearly every deep learning framework that's currently available. This simply a much easier way to go and takes a lot less time. They also similar deep learning AMIs that have source code on them, so if you really want to you can build out your framework yourself. Additionally, I discovered a few other tweaks that made using Jupyter easier as well, making sure the gpu is firing on all cylinders. When it's all said and done, I've found speed-ups training CNNs of 5 to 10 times over my little gpu on my windows machine and even more over my laptop.  

---

### 1 - Sign-up/in.  
<div style="text-align:center;"><img src="/assets/aws-create-account.png"></div>  

Visit [aws.amazon.com](https://aws.amazon.com/) and sign-up for a free basic support plan. Setting up [AWS EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html) for Linux instances is what this guide was created for and it takes a bit of time as there's a quite a few details for configuring your account, users and access. In the navigation pane for the Amazon Elastic Compute Cloud Documentation I recommend at minimum following the directions in:

- [**Setting Up**](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/get-set-up-for-amazon-ec2.html)
- [**Getting Started**](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)
- [**Best Practices**](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-best-practices.html)
- [**Instances**](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Instances.html)
- [**Network and Security**](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_Network_and_Security.html)  

 before continuing with this guide if you haven't already.  

---

### 2 - Change Region.  
<div style="text-align:center;"><img src="/assets/ec2-region.png"></div>  

Once you're signed into your account, check to see which region you're in. You should change it to whichever is closest to you, e.g. this guide is set to [US East(Ohio) (us-east-2)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-available-regions).  

---

### 3 - Request Limit Increase.  
<div style="text-align:center;"><img src="/assets/ec2-limits.png"></div>  

In your EC2 Management Console, navigate to your EC2 Dashboard. In the navigation pane on the left side of the console, choose **Limits**.  In the EC2 Service Limits report and find your "Current Limit" for the p2.xlarge instance type. If your limit of p2.xlarge instances is 0, you'll need to increase the limit before you can launch an instance. From the EC2 Service Limits page, click on “Request limit increase” next to “p2.xlarge”.

You will not be charged for requesting a limit increase. You will only be charged once you actually launch an instance.  

<div style="text-align:center;"><img src="/assets/ec2-limit-increase.png"></div>  

<div style="text-align:center;"><img src="/assets/ec2-limit-increase2.png"></div>  

**Wait for approval.** It can take up to 48 hours to receive approval for your request.

---

### 4 - Launch An Instance.  

<div style="text-align:center;"><img src="/assets/ec2-panel.png"></div>  

Once AWS approves your GPU Limit Increase Request, you can start the process of launching your instance.

In the navigation pane on the left side of the console, choose **Instances**, and click on the **Launch Instance** button.

Back in the navigation pane, choose **AWS Marketplace** and search for "deep learning ami ubuntu" and select it:  

<div style="text-align:center;"><img src="/assets/ec2-ami.png"></div>  

After selecting the ami, you'll be shown a page cataloguing the different charges for using this instance. For GPU compute extra large (p2.xlarge) it just shows that you will be charged $0.90/hr, which is their standard rate. Click continue.

---  

### 5 - Select Instance Type  

<div style="text-align:center;"><img src="/assets/ec2-instance-type.png"></div>  

In "Step 2: Choose an Instance Type", using the "Filter by:" drop-down menu select GPU compute and select the p2.xlarge GPU instance type.  

---  

### 6 - Configure Security Group  

<div style="text-align:center;"><img src="/assets/ec2-security-group.png"></div>  

In the header, select **6. Configure Security Group**. Under the **Type** column click the **Add Rule** drop-down menu and select "Custom TCP". In the **Port Range** field type `8888`. And in the drop-down menu under **Source** select "Anywhere". Finally, at the bottom of the page click **Review and Launch**.  

---  

### 7 - Launch Your Configured Instance  

<div style="text-align:center;"><img src="/assets/ec2-launch.png"></div>  

When you click **Launch**, under **Select a key pair** choose your key and click **Launch Instance**.  

---  

### 8 - Connect To Your Running Instance  

<div style="text-align:center;"><img src="/assets/ec2-running.png"></div>  

Now that your instance is up and running you need to connect to it. Since we'll be running Jupyter Notebooks in this instance you'll want to port-forward the ec2 instance to your local machine. Refer to AWS documentation for [Configure the Client to Connect to the Jupyter Server](https://docs.aws.amazon.com/dlami/latest/devguide/setup-jupyter-configure-client.html) for connection instructions for different OS's.  

The screenshot below shows how to do it for OSX:

<div style="text-align:center;"><img src="/assets/ec2-configure-client.png"></div>  

---  

### 9 - Optimize The Instance GPU  

<div style="text-align:center;"><img src="/assets/ec2-timer.png"></div>  

Once you've connected, you now need to optimize the GPU. Run the following code in the terminal that is connected to your instance:

```
wget "https://gist.github.com/m00nd00r/d52df1c881b0f294fa8baa98dbcf01cb/archive/977bffb50b3c4f3acaadadc418d7c1b9d1d9b80b.zip" -O temp.zip; unzip -j temp.zip; rm temp.zip; chmod 555 nvidia-setup.sh; ./nvidia-setup.sh
```  

This will download a script I wrote to execute several nvidia commands to maximize the GPU performance that is hosted in a gist I created for it. It then executes the script. This script runs a timer to clock how fast the nvidia-smi utility runs. You should see output time to be less than 1 second.  

### 10 - Configure Jupyter  

Now that your instance's GPU is optimized, you'll need to configure jupyter to run a little more easily.
reference: https://docs.aws.amazon.com/dlami/latest/devguide/setup-jupyter-config.html

Run the following code in the terminal that is connected to your instance:
```
wget "https://gist.github.com/m00nd00r/25657aa82f968d5ebe82fd5f33a55bc5/archive/86de1b1867c68e089b5647555dcfdb2fec9029e8.zip" -O temp.zip; unzip -j temp.zip; rm temp.zip; chmod 555 jupyter-setup.sh; ./jupyter-setup.sh
```

Similar to optimizing the GPU, the above code downloads, unzips and runs a script that will set-up your jupyter config file. When you execute the above line of code in your instance it will prompt you to create a password to login to jupyter. It will then ask you to verify the password. Either enter a password or just press enter if you don't want one. Either way you'll need to enter the same at the Jupyter login page. Then wait for it to execute - don't type anything, it takes a moment.

<div style="text-align:center;"><img src="/assets/ec2-config1.png"></div>  

It will then ask you if you want overwrite the default config file. Type y and enter. It will finish configuring jupyter for you. Then you're ready open a notebook and get started.

<div style="text-align:center;"><img src="/assets/ec2-config2.png"></div>  

### 11 - Start A Notebook

Simply type in your instance terminal window:

`jupyter notebook`

and it will run through it's normal sequence.

<div style="text-align:center;"><img src="/assets/ec2-run-notebook.png"></div>  

When you see the screen above, type https://127.0.0.1:8157 into your browser. Initially you'll see the following warning:  

<div style="text-align:center;"><img src="/assets/ec2-connect1.png"></div>  

Click the "Advanced" button and you will see:  

<div style="text-align:center;"><img src="/assets/ec2-connect2.png"></div>  

Click "Add Exception..." and you will see:  

<div style="text-align:center;"><img src="/assets/ec2-connect3.png"></div>  

Click "Confirm Securtiy Exception" and you will see the jupyter login page:  

<div style="text-align:center;"><img src="/assets/ec2-connect4.png"></div>  

Now enter your password or press enter and that's it! You're in!  

Happy Computing!
