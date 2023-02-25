import { touchableStyles} from "../../styles/css/touchableStyles";
import { useModalState } from '@rainbow-me/rainbowkit/dist/components/RainbowKitProvider/ModalContext';
import { useEffect, useState } from 'react';
import { Box } from './Box';
import { Dialog } from './Dialog';
import { DialogContent } from './DialogContent';
import { useModalStateValue } from './ModalContext';
import { useSendTransaction, usePrepareSendTransaction, useAccount } from 'wagmi';
import { BigNumber, ethers } from 'ethers';
import { formatUnits, parseUnits } from 'ethers/lib/utils';
import { NFTStorage } from 'nft.storage';
// import { Dialog } from './Dialog';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { useRouter } from 'next/router';
import { useSelloutModal} from "../../context/EventOutProvider";

type ItemMetaData = {
    title: string;
    price: number;
    image: string;
    description: string;
};

export default function SellOutCheckOut({
                                            itemMetaData,
                                            handleMint,
                                            address,
                                            promoCode,
                                        }: {
    itemMetaData: ItemMetaData;
    handleMint: () => void;
    address: boolean;
    promoCode: boolean;
}) {
    const { closeModal, isModalOpen } = useModalStateValue();
    const { modalType, modalTheme, setModalAccent, setModalTheme } = useSelloutModal();
    const { title, price, description, image } = itemMetaData;
    const [totalPrice, setTotalPrice] = useState(0.0);
    const [shippingPrice, setShippingPrice] = useState(0.1);
    return (
        <div data-theme={modalTheme} className="p-10 bg-bgmain">
    <div className="flex items-center flex-col">
    <h1 className="text-3xl font-bold mb-5 font-rounded text-textmain">Checkout</h1>
        <ItemMetaData title={title} price={price} description={description} image={image} />
    {modalType === 'PRODUCT' && (
        <OrderSummary price={price} shipping={shippingPrice} setTotalPrice={setTotalPrice} />
    )}
    {promoCode && <PromoCode />}
    {address && <AddressInput />}
    <div className="mt-5">
    <PaymentButton
        handleMint={handleMint}
    totalPrice={totalPrice}
    itemMetaData={itemMetaData}
    shippingPrice={shippingPrice}
    />
    </div>
    </div>
    <ThemeModes />
    </div>
);
}

export function ItemMetaData({
                                 title,
                                 price,
                                 image,
                                 description,
                             }: {
    title: string;
    price: number;
    image: string;
    description: string;
}) {
    const { modalType } = useSelloutModal();
    return (
        <div className=" rounded-2xl w-full flex-row mx-10  flex h-36 shadow-lg">
        <div className=" flex flex-1 items-center justify-center">
        <img src={image} className="w-28 h-auto  rounded-xl" />
        </div>
        <div className=" flex-col justify-evenly  text-textmain font-rounded   flex flex-[1.6] p-2 ">
        <h1>{title}</h1>
    {modalType === 'PRODUCT' ? <h1>{price} ETH</h1> : <h1 className="italic">Cost of gas</h1>}
    </div>
    </div>
);
}

export function OrderSummary({
                                 price,
                                 shipping,
                                 setTotalPrice,
                             }: {
    price: number;
    shipping: number;
    setTotalPrice: any;
}) {
    const totalPrice = price + shipping;
    useEffect(() => {
        setTotalPrice(totalPrice);
    }, [setTotalPrice, totalPrice]);
    return (
        <div className=" rounded-2xl text-textmain w-full flex-col mx-10 flex h-52 shadow mt-10 p-5">
        <div className="border-b border-b-gray-200 pb-6">Order Summary</div>
    <div className="flex mt-5 justify-between">
        <div>Order</div>
        <div>{price} ETH</div>
    </div>
    <div className="flex mt-5 justify-between">
        <div>Shipping</div>
        <div>{shipping} ETH</div>
    </div>
    <div className="flex mt-5 justify-between">
        <div>Total</div>
        <div>{totalPrice.toFixed(4)} ETH</div>
    </div>
    </div>
);
}

async function getExampleImage(image) {
    const r = await fetch(image);
    if (!r.ok) {
        throw new Error(`error fetching image: [${r.statusCode}]: ${r.status}`);
    }
    return r.blob();
}

async function storeExampleNFT(
    itemMetaData: ItemMetaData,
    txHash: string,
    account: string,
    totalPrice: number,
    shippingPrice: number,
) {
    const image = await getExampleImage(itemMetaData.image);
    const name = itemMetaData.title;
    const nft = {
        image, // use image Blob as `image` field
        name: name,
        description: '',
        properties: {
            type: 't-shirt purchase',
            origins: {
                txHash: txHash,
                price: itemMetaData.price,
                orderDate: new Date().toLocaleDateString(),
                shippingPrice: shippingPrice,
                totalPrice: totalPrice,
            },
            authors: [{ account: account }],
            content: {
                'text/markdown': 'This is an example of a purhase of a tshirt using sellout',
            },
        },
    };

    const client = new NFTStorage({ token: process.env.NFT_STORAGE_KEY });
    const metadata = await client.store(nft);

    // console.log('NFT data stored!');
    // console.log('Metadata URI: ', metadata);
    return metadata.ipnft;
}

function getPaymentButtonLabel(ipfsHash, modalType) {
    if (ipfsHash) return 'View Receipt';
    else {
        switch (modalType) {
            case 'NFT':
                return 'Mint';
            case 'PRODUCT':
                return 'Confirm Payment';
            default:
                return 'Confirm Payment';
        }
    }
}

export function PaymentButton({
                                  totalPrice,
                                  itemMetaData,
                                  shippingPrice,
                                  handleMint,
                              }: {
    totalPrice: number;
    itemMetaData: ItemMetaData;
    shippingPrice: number;
    handleMint: () => void;
}) {
    const weiPrice = totalPrice && parseUnits(totalPrice.toString());
    const [ipfsHash, setIpfsHash] = useState('');
    const { address } = useAccount();
    const { modalType } = useSelloutModal();
    console.log(modalType, 'modalType');
    // const viewReceipt = () => toast('View Receipt');
    const notify = () => toast('Wow so easy !');
    const { push } = useRouter();
    const { config } = usePrepareSendTransaction({
        request: {
            to: '0xE35ef95A80839C3c261197B6c93E5765C9A6a31a',
            value: weiPrice,
        },
    });
    const { data, isLoading, isSuccess, sendTransaction } = useSendTransaction(config);

    const storeExampleNFTAsync = async (itemMetaData, data, address) => {
        const ipfsUrl = await storeExampleNFT(itemMetaData, data?.hash, address, totalPrice, shippingPrice);
        console.log('ipfsUrl', ipfsUrl);
        setIpfsHash(ipfsUrl);
    };

    // useEffect(() => {
    // 	if (ipfsHash) {
    // 		console.log('ipfsHashNew', ipfsHash);
    // 		notify();
    // 	}
    // }, [ipfsHash]);

    useEffect(() => {
        if (isSuccess && data && address) {
            storeExampleNFTAsync(itemMetaData, data, address);
        }
    }, [isSuccess, data]);

    return (
        <>
            <Box
                as="button"
    background="accentColor"
    borderRadius="connectButton"
    boxShadow="connectButton"
    className={touchableStyles({ active: 'shrink', hover: 'grow' })}
    color="accentColorForeground"
    fontFamily="body"
    fontWeight="bold"
    height="40"
    paddingLeft="36"
    paddingRight="36"
    key="connect"
    onClick={(e) => {
        if (modalType === 'NFT') {
            handleMint();
        } else {
            if (ipfsHash) {
                push(`/receipt/${ipfsHash}`);
            } else {
                e.stopPropagation();
                sendTransaction?.();
            }
        }
    }}
    paddingX="14"
    transition="default"
    type="button"
        >
        {getPaymentButtonLabel(ipfsHash, modalType)}
        </Box>
        </>
);
}

export function ThemeModes() {
    const { setModalTheme } = useSelloutModal();
    return (
        <div className="mb-2">
            {/* <span className="pb-2 text-textmain font-rounded ">Mode:</span> */}
            <div className="flex flex-row">
        {[
                { theme: 'DARK', color: '#22272D' },
    { theme: 'SLATE', color: '#1A1A1E' },
    { theme: 'LIGHT', color: '#FFFEFE' },
].map((theme, i) => {
        return (
            <div
                onClick={(e) => {
            e.stopPropagation();
            setModalTheme(theme.theme);
        }}
        key={i}
        style={{ backgroundColor: theme.color }}
        className={`z-10 rounded-full mt-2 w-5 h-5 ${i !== 0 && 'ml-2'}  border`}
        />
    );
    })}
    </div>
    </div>
);
}

export function AddressInput() {
    return (
        <div
            className=" rounded-2xl text-textmain w-full flex-col mx-10 flex h-28 shadow mt-5 p-5"
    onClick={(e) => {
        e.stopPropagation();
    }}
>
    <input
        onFocus={(e) => {
        e.stopPropagation();
    }}
    type="description"
    placeholder="Enter your address"
    className="bg-bgmain h-full placeholder:text-textmain"
        />
        </div>
);
}

export function PromoCode() {
    return (
        <div
            className=" rounded-2xl text-textmain w-full flex-col mx-10 flex h-16 shadow mt-5 p-5"
    onClick={(e) => {
        e.stopPropagation();
    }}
>
    <input
        onFocus={(e) => {
        e.stopPropagation();
    }}
    type="text"
    className="bg-bgmain placeholder:text-textmain"
    placeholder="Enter your promo code"
        />
        </div>
);
}
